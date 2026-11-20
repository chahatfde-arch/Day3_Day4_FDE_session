"""Related parties: the question a table cannot answer.

The one idea this module teaches: **some questions are about the shape of the links, not about
the rows.** Good Bank's policy section 5.1 says a group is "any set of companies linked by a
common director or a common registered address, directly or through a chain of such links".

"Through a chain" is the hard part. In SQL you do not know in advance how many joins you need,
so you end up writing a recursive CTE and hoping. In a graph you say "follow this kind of edge
one to eight times" and the database does the walking:

    MATCH (c:Company {company_id: $company_id})-[:DIRECTOR_OF|REGISTERED_AT*1..8]-(m:Company)

The data model is three node labels and two edge types:

    (:Director)-[:DIRECTOR_OF]->(:Company)-[:REGISTERED_AT]->(:Address)

Two companies are in the same group when an undirected walk connects them: company, up to its
director, back down to the other company (a shared director), or company, out to its address,
back to the other company (a shared address). Each company hop is two edges, which is why the
Cypher divides the path length by two.

Backends: NetworkX in memory by default, Neo4j when `NEO4J_URI` is set (AuraDB Free or Docker).
The Cypher text is recorded on every `GroupExposure` **even when NetworkX did the work**, so an
auditor sees the same query in the memo no matter which backend ran, and so a beginner can copy
it straight into the Neo4j Browser.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from creditlens.config import settings
from creditlens.generators.policy import RULES
from creditlens.observability import observe
from creditlens.schemas import GroupExposure, GroupMember

log = logging.getLogger(__name__)

MAX_HOPS = 8  # policy 5.1 says "a chain of such links"; 8 is far deeper than the data ever goes

# ---------------------------------------------------------------------------
# The Cypher. Kept as module constants so the audit trail, the docs and the Neo4j
# backend all use the exact same text, and it can never drift between them.
# ---------------------------------------------------------------------------

CYPHER_CONSTRAINTS = """
CREATE CONSTRAINT company_id IF NOT EXISTS
FOR (c:Company) REQUIRE c.company_id IS UNIQUE;
CREATE CONSTRAINT director_id IF NOT EXISTS
FOR (d:Director) REQUIRE d.director_id IS UNIQUE;
CREATE CONSTRAINT address_value IF NOT EXISTS
FOR (a:Address) REQUIRE a.value IS UNIQUE;
""".strip()

CYPHER_LOAD_COMPANIES = """
UNWIND $rows AS row
MERGE (c:Company {company_id: row.company_id})
SET c.legal_name = row.legal_name,
    c.sector = row.sector,
    c.region = row.region,
    c.outstanding_inr_000 = row.outstanding_inr_000,
    c.pending_request_inr_000 = row.pending_request_inr_000
MERGE (a:Address {value: row.registered_address})
MERGE (c)-[:REGISTERED_AT]->(a)
""".strip()

CYPHER_LOAD_DIRECTORS = """
UNWIND $rows AS row
MERGE (d:Director {director_id: row.director_id})
SET d.full_name = row.full_name
WITH d, row
MATCH (c:Company {company_id: row.company_id})
MERGE (d)-[:DIRECTOR_OF {role: row.role}]->(c)
""".strip()

CYPHER_RELATED_GROUP = f"""
MATCH (c:Company {{company_id: $company_id}})
MATCH path = (c)-[:DIRECTOR_OF|REGISTERED_AT*1..{MAX_HOPS * 2}]-(m:Company)
WHERE m.company_id <> c.company_id
WITH m, min(length(path)) / 2 AS hops
RETURN m.company_id AS company_id,
       m.legal_name AS legal_name,
       hops,
       m.outstanding_inr_000 AS existing_outstanding_inr_000,
       m.pending_request_inr_000 AS pending_request_inr_000
ORDER BY hops, company_id
""".strip()

CYPHER_GROUP_EXPOSURE = f"""
MATCH (c:Company {{company_id: $company_id}})
OPTIONAL MATCH (c)-[:DIRECTOR_OF|REGISTERED_AT*1..{MAX_HOPS * 2}]-(m:Company)
WITH collect(DISTINCT m) + collect(DISTINCT c) AS all_members
UNWIND all_members AS member
WITH DISTINCT member
RETURN count(member) AS members,
       sum(coalesce(member.outstanding_inr_000, 0)) AS existing_inr_000,
       sum(coalesce(member.pending_request_inr_000, 0)) AS pending_inr_000,
       sum(coalesce(member.outstanding_inr_000, 0)
           + coalesce(member.pending_request_inr_000, 0)) AS total_exposure_inr_000
""".strip()

CYPHER_ALL_GROUPS = f"""
MATCH (c:Company)-[:DIRECTOR_OF|REGISTERED_AT*1..{MAX_HOPS * 2}]-(m:Company)
WHERE c.company_id < m.company_id
RETURN c.company_id AS a, m.company_id AS b
""".strip()

CYPHER_LINK_REASON = """
MATCH (a:Company {company_id: $a})<-[:DIRECTOR_OF]-(d:Director)-[:DIRECTOR_OF]->(b:Company {company_id: $b})
RETURN 'shared_director' AS link, d.director_id AS detail
UNION
MATCH (a:Company {company_id: $a})-[:REGISTERED_AT]->(ad:Address)<-[:REGISTERED_AT]-(b:Company {company_id: $b})
RETURN 'shared_address' AS link, ad.value AS detail
""".strip()

CYPHER_PROFILE = """
PROFILE
MATCH (c:Company {company_id: 'C031'})-[:DIRECTOR_OF|REGISTERED_AT*1..16]-(m:Company)
RETURN DISTINCT m.company_id
""".strip()


def cypher_cheatsheet() -> dict[str, str]:
    """Every query this module uses, named, so the docs and the audit trail stay in sync."""
    return {
        "constraints": CYPHER_CONSTRAINTS,
        "load_companies": CYPHER_LOAD_COMPANIES,
        "load_directors": CYPHER_LOAD_DIRECTORS,
        "related_group": CYPHER_RELATED_GROUP,
        "group_exposure": CYPHER_GROUP_EXPOSURE,
        "all_groups": CYPHER_ALL_GROUPS,
        "link_reason": CYPHER_LINK_REASON,
        "profile": CYPHER_PROFILE,
    }


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class GraphStore:
    """Companies, directors and addresses as a graph, with group queries on top.

    Typical use:

        store = GraphStore().load_from_registry(settings.paths.raw)
        exposure = store.group_exposure("C031")
        if exposure.breaches_limit:
            ...  # policy rule 5.2 is an exception; file a Jira ticket

    Args:
        backend: "networkx" or "neo4j". Defaults to "neo4j" when `NEO4J_URI` is set.
        limit_inr_000: the policy 5.2 limit. Defaults to `RULES["MAX_GROUP_EXPOSURE_INR_000"]`.
    """

    def __init__(self, backend: Optional[str] = None,
                 limit_inr_000: Optional[int] = None) -> None:
        configured = settings.neo4j.get("uri")
        self.backend = backend or ("neo4j" if configured else "networkx")
        self.limit_inr_000 = int(limit_inr_000 if limit_inr_000 is not None
                                 else RULES["MAX_GROUP_EXPOSURE_INR_000"])
        self.graph = None           # networkx.Graph, built by load_from_registry
        self._driver = None         # neo4j.Driver when the Neo4j backend is used
        self.companies: dict[str, dict] = {}
        self.outstanding: dict[str, int] = {}
        self.pending: dict[str, int] = {}

    # -- loading -----------------------------------------------------------

    @observe(name="graph.load_from_registry")
    def load_from_registry(self, raw_dir: Path) -> "GraphStore":
        """Read the CSVs under `data/raw` and build the graph.

        `raw_dir` is the `data/raw` folder (the registry CSVs live in `raw_dir/registry` and
        the applications in `raw_dir/applications`). Passing the registry folder itself also
        works, because that is what a beginner tries first.
        """
        raw_dir = Path(raw_dir)
        registry = raw_dir / "registry" if (raw_dir / "registry").exists() else raw_dir
        companies = pd.read_csv(registry / "companies.csv")
        links = pd.read_csv(registry / "company_directors.csv")
        directors = _read_optional(registry / "directors.csv")
        existing = _read_optional(registry / "existing_loans.csv")
        applications = _read_optional(_applications_path(raw_dir, registry))

        self.companies = {row["company_id"]: row for row in companies.to_dict("records")}
        self.outstanding = ({r["company_id"]: int(r["outstanding_inr_000"])
                             for r in existing.to_dict("records")} if existing is not None else {})
        self.pending = ({cid: int(total) for cid, total in
                         applications.groupby("company_id")["requested_inr_000"].sum().items()}
                        if applications is not None else {})
        # Director names are not PII-free (see CONTRACTS "PII"), so only the id reaches a prompt.
        director_names = ({r["director_id"]: r["full_name"] for r in directors.to_dict("records")}
                          if directors is not None else {})

        self._build_networkx(companies, links, director_names)
        if self.backend == "neo4j":
            self._load_neo4j(companies, links, director_names)
        return self

    def _build_networkx(self, companies: pd.DataFrame, links: pd.DataFrame,
                        director_names: dict) -> None:
        import networkx as nx

        g = nx.Graph()
        for row in companies.to_dict("records"):
            cid = row["company_id"]
            g.add_node(_company_node(cid), kind="Company", company_id=cid,
                       legal_name=row["legal_name"],
                       outstanding_inr_000=self.outstanding.get(cid, 0),
                       pending_request_inr_000=self.pending.get(cid, 0))
            address = str(row["registered_address"]).strip()
            g.add_node(_address_node(address), kind="Address", value=address)
            # REGISTERED_AT: company -> address. NetworkX keeps it undirected because every
            # group query walks it in both directions anyway.
            g.add_edge(_company_node(cid), _address_node(address), kind="REGISTERED_AT")
        for row in links.to_dict("records"):
            did, cid = row["director_id"], row["company_id"]
            if _company_node(cid) not in g:
                continue
            g.add_node(_director_node(did), kind="Director", director_id=did,
                       full_name=director_names.get(did, ""))
            # DIRECTOR_OF: director -> company.
            g.add_edge(_director_node(did), _company_node(cid), kind="DIRECTOR_OF",
                       role=row.get("role", ""))
        self.graph = g

    def _load_neo4j(self, companies: pd.DataFrame, links: pd.DataFrame,
                    director_names: dict) -> None:
        """MERGE the same data into Neo4j. MERGE, not CREATE, so re-running is harmless."""
        driver = self._neo4j_driver()
        if driver is None:
            log.warning("NEO4J_URI is set but the driver could not connect. Using NetworkX.")
            self.backend = "networkx"
            return
        company_rows = [{
            "company_id": r["company_id"], "legal_name": r["legal_name"],
            "sector": r.get("sector", ""), "region": r.get("region", ""),
            "registered_address": str(r["registered_address"]).strip(),
            "outstanding_inr_000": self.outstanding.get(r["company_id"], 0),
            "pending_request_inr_000": self.pending.get(r["company_id"], 0),
        } for r in companies.to_dict("records")]
        director_rows = [{
            "director_id": r["director_id"], "company_id": r["company_id"],
            "full_name": director_names.get(r["director_id"], ""), "role": r.get("role", ""),
        } for r in links.to_dict("records")]
        with driver.session() as session:
            for statement in CYPHER_CONSTRAINTS.split(";"):
                if statement.strip():
                    session.run(statement)
            session.run(CYPHER_LOAD_COMPANIES, rows=company_rows)
            session.run(CYPHER_LOAD_DIRECTORS, rows=director_rows)

    def _neo4j_driver(self):
        if self._driver is not None:
            return self._driver
        config = settings.neo4j
        if not config.get("uri"):
            return None
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(config["uri"],
                                          auth=(config["user"], config["password"]))
            driver.verify_connectivity()
            self._driver = driver
        except Exception as exc:  # noqa: BLE001  no database is a fallback, not a crash
            log.warning("Neo4j unavailable (%s). Falling back to NetworkX.", exc)
            self._driver = None
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # -- the group queries -------------------------------------------------

    @observe(name="graph.related_group")
    def related_group(self, company_id: str) -> list[GroupMember]:
        """Every company reachable from `company_id` through shared directors or addresses.

        The applicant itself is always the first member, with `link="self"` and `hops=0`.
        Other members carry how they were first reached and at how many company hops.
        """
        self._require_graph()
        if company_id not in self.companies:
            raise KeyError(f"unknown company_id: {company_id}")
        if self.backend == "neo4j" and self._neo4j_driver() is not None:
            return self._related_group_neo4j(company_id)
        return self._related_group_networkx(company_id)

    def _related_group_networkx(self, company_id: str) -> list[GroupMember]:
        """Breadth-first search over company-to-company links, remembering why and how far.

        BFS (not depth-first) because we want the *shortest* chain of links to each member:
        "two hops away through a shared address" is the fact an underwriter asks about.
        """
        members = [self._member(company_id, "self", 0)]
        seen = {company_id}
        queue: deque[tuple[str, int]] = deque([(company_id, 0)])
        while queue:
            current, hops = queue.popleft()
            if hops >= MAX_HOPS:
                continue
            for neighbour, link in self._company_neighbours(current):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                members.append(self._member(neighbour, link, hops + 1))
                queue.append((neighbour, hops + 1))
        return members

    def _company_neighbours(self, company_id: str) -> Iterable[tuple[str, str]]:
        """Companies one company-hop away: via a shared director, or a shared address."""
        g = self.graph
        node = _company_node(company_id)
        for middle in g.neighbors(node):
            kind = g.nodes[middle]["kind"]
            link = "shared_director" if kind == "Director" else "shared_address"
            for other in g.neighbors(middle):
                if other == node or g.nodes[other]["kind"] != "Company":
                    continue
                yield g.nodes[other]["company_id"], link

    def _related_group_neo4j(self, company_id: str) -> list[GroupMember]:
        members = [self._member(company_id, "self", 0)]
        with self._driver.session() as session:
            for record in session.run(CYPHER_RELATED_GROUP, company_id=company_id):
                link = self._link_reason_neo4j(company_id, record["company_id"]) \
                    if record["hops"] == 1 else "chain"
                members.append(self._member(record["company_id"], link, int(record["hops"])))
        return members

    def _link_reason_neo4j(self, a: str, b: str) -> str:
        with self._driver.session() as session:
            record = session.run(CYPHER_LINK_REASON, a=a, b=b).single()
            return record["link"] if record else "chain"

    def _member(self, company_id: str, link: str, hops: int) -> GroupMember:
        row = self.companies.get(company_id, {})
        return GroupMember(
            company_id=company_id,
            legal_name=str(row.get("legal_name", company_id)),
            link=link,
            hops=hops,
            existing_outstanding_inr_000=int(self.outstanding.get(company_id, 0)),
            pending_request_inr_000=int(self.pending.get(company_id, 0)),
        )

    @observe(name="graph.group_exposure")
    def group_exposure(self, company_id: str, requested: Optional[dict] = None) -> GroupExposure:
        """Total what the group already owes plus what it is asking for, and test policy 5.2.

        Args:
            company_id: the applicant.
            requested: optional `{company_id: amount_inr_000}` that replaces the pending
                request read from `applications.csv` for those companies. Use it to answer
                "what if we also approved this one?" without touching the data.

        Returns:
            A `GroupExposure` listing every member (policy 5.3 requires the full list even when
            the limit is not breached), the total, the limit and the Cypher that defines it.
        """
        members = self.related_group(company_id)
        if requested:
            members = [m.model_copy(update={"pending_request_inr_000": int(requested[m.company_id])})
                       if m.company_id in requested else m for m in members]
        total = sum(m.existing_outstanding_inr_000 + m.pending_request_inr_000 for m in members)
        ids = sorted(m.company_id for m in members)
        return GroupExposure(
            company_id=company_id,
            # A deterministic label, not the instructor answer key's G1..G4: the graph has no
            # way to know the order the data was generated in.
            group_id=f"GRP-{ids[0]}" if len(ids) > 1 else None,
            members=members,
            total_exposure_inr_000=int(total),
            limit_inr_000=self.limit_inr_000,
            # Policy 5.1 defines a group as a *set* of linked companies, and 5.2 limits
            # exposure "to a group". A company with no related parties is therefore not a
            # 5.2 exception however large its own facility is; that is rule 3, not rule 5.
            # The total is still reported, so an analyst always sees the number.
            breaches_limit=bool(len(members) > 1 and total > self.limit_inr_000),
            cypher=self._audit_cypher(company_id),
        )

    def _audit_cypher(self, company_id: str) -> str:
        """The query an auditor can paste into the Neo4j Browser to reproduce this answer."""
        backend_note = ("// Executed on Neo4j." if self.backend == "neo4j"
                        else "// Executed in NetworkX; this is the equivalent Cypher, recorded "
                             "so the audit trail does not depend on which backend ran.")
        return (f"{backend_note}\n// :param company_id => '{company_id}'\n\n"
                f"// 1. the group\n{CYPHER_RELATED_GROUP}\n\n"
                f"// 2. the exposure\n{CYPHER_GROUP_EXPOSURE}")

    @observe(name="graph.all_groups")
    def all_groups(self) -> list[set[str]]:
        """Every related-party group in the book: connected components with more than one company.

        This is the Day 3 "find what the data was hiding" moment. Nothing in the CSVs says
        "these four companies are a group"; the graph derives it from the links alone.
        """
        self._require_graph()
        if self.backend == "neo4j" and self._neo4j_driver() is not None:
            return self._all_groups_neo4j()
        seen: set[str] = set()
        groups: list[set[str]] = []
        for company_id in self.companies:
            if company_id in seen:
                continue
            component = {m.company_id for m in self._related_group_networkx(company_id)}
            seen |= component
            if len(component) > 1:
                groups.append(component)
        return sorted(groups, key=lambda g: sorted(g))

    def _all_groups_neo4j(self) -> list[set[str]]:
        """Pull the company-to-company pairs out of Neo4j and union them into components."""
        pairs: list[tuple[str, str]] = []
        with self._driver.session() as session:
            for record in session.run(CYPHER_ALL_GROUPS):
                pairs.append((record["a"], record["b"]))
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        groups: dict[str, set[str]] = {}
        for node in list(parent):
            groups.setdefault(find(node), set()).add(node)
        return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: sorted(g))

    # -- Day 3: profile and optimize ---------------------------------------

    def profile(self, company_id: Optional[str] = None, repeats: int = 20) -> dict:
        """Time the NetworkX traversal and explain how to do the same on Neo4j.

        There is no profiler for a dict of sets, so for NetworkX "profiling" means timing. The
        interesting part is the advice for the real database, which is in the returned dict and
        printed by `python -m creditlens.retrieval.graph --profile`.
        """
        self._require_graph()
        company_id = company_id or next(iter(self.companies))
        start = time.perf_counter()
        for _ in range(repeats):
            members = self._related_group_networkx(company_id)
        elapsed = time.perf_counter() - start
        return {
            "backend": self.backend,
            "company_id": company_id,
            "companies": len(self.companies),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "group_size": len(members),
            "repeats": repeats,
            "avg_ms": round(elapsed / repeats * 1000, 3),
            "neo4j_notes": [
                "Run the query with EXPLAIN first: it shows the plan without touching the data, "
                "so a bad plan costs nothing to discover.",
                "Then run it with PROFILE: it executes and reports db hits per operator. The "
                "operator with the most db hits is the one to fix; everything else is noise.",
                "The usual first finding is `NodeByLabelScan` on Company. That means the "
                "database read every company to find one. Create a uniqueness constraint "
                "(CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE "
                "c.company_id IS UNIQUE), which also creates the index, and the operator "
                "becomes `NodeUniqueIndexSeek` with a handful of db hits.",
                "Bound the variable-length pattern. `*1..16` is a promise to the planner that "
                "the walk stops; an unbounded `*` on a dense graph can expand astonishingly far.",
                "Return DISTINCT company ids rather than whole paths. The number of paths "
                "through a group grows much faster than the number of members in it.",
            ],
            "cypher_profile": CYPHER_PROFILE,
        }

    # -- helpers -----------------------------------------------------------

    def _require_graph(self) -> None:
        if self.graph is None:
            raise RuntimeError("call load_from_registry(raw_dir) before querying the graph")

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        n = self.graph.number_of_nodes() if self.graph is not None else 0
        return f"GraphStore(backend={self.backend!r}, companies={len(self.companies)}, nodes={n})"


def _company_node(company_id: str) -> str:
    return f"Company:{company_id}"


def _director_node(director_id: str) -> str:
    return f"Director:{director_id}"


def _address_node(address: str) -> str:
    return f"Address:{address}"


def _applications_path(raw_dir: Path, registry: Path) -> Path:
    candidate = raw_dir / "applications" / "applications.csv"
    if candidate.exists():
        return candidate
    return registry.parent / "applications" / "applications.csv"


def _read_optional(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV if it is there. A missing file means "that number is zero", not a crash."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None


def main() -> None:  # pragma: no cover - a convenience CLI, not part of the contract
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Related-party groups from the company registry")
    parser.add_argument("company_id", nargs="?", help="show the group and exposure for this company")
    parser.add_argument("--profile", action="store_true", help="time the traversal and print tuning notes")
    parser.add_argument("--cheatsheet", action="store_true", help="print every Cypher query used")
    args = parser.parse_args()

    if args.cheatsheet:
        for name, query in cypher_cheatsheet().items():
            print(f"\n-- {name} --\n{query}")
        return

    store = GraphStore().load_from_registry(settings.paths.raw)
    if args.profile:
        print(json.dumps(store.profile(args.company_id), indent=2))
        return
    if args.company_id:
        exposure = store.group_exposure(args.company_id)
        print(exposure.model_dump_json(indent=2))
        return
    for group in store.all_groups():
        print(sorted(group))


if __name__ == "__main__":  # pragma: no cover
    main()
