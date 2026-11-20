"""Central configuration for CreditLens AI.

Every generated dataset depends only on SEED and the sizes below, so the whole
data folder can be deleted and rebuilt identically at any time.

Secrets (API keys, database passwords) are read from the environment or from a
local .env file. Nothing secret lives in this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader so the project needs no extra dependency."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def clean(self) -> Path:
        return self.data / "clean"

    @property
    def curated(self) -> Path:
        return self.data / "curated"

    @property
    def real(self) -> Path:
        """Small samples of real public datasets (Kaggle, Hugging Face). See creditlens.data_real."""
        return self.data / "real"

    @property
    def answer_key(self) -> Path:
        """Instructor-only truth. Never give this folder to workshop pods."""
        return self.data / "answer_key"

    @property
    def cache(self) -> Path:
        """LLM response cache, embeddings and other re-computable files."""
        return self.data / "cache"

    @property
    def ground_truth(self) -> Path:
        return self.root / "ground_truth"

    def with_data_root(self, data_root: Path) -> "Paths":
        return Paths(root=self.root, data=Path(data_root))


@dataclass(frozen=True)
class Sizes:
    companies: int = 40
    applications: int = 30
    statement_months: int = 12
    related_party_groups: int = 4
    scanned_financials_share: float = 0.25
    statement_pdfs: int = 6
    loan_history_rows: int = 2000
    # Real-data samples. Small on purpose: the workshop is live and bandwidth is shared.
    real_receipts: int = 40          # SROIE scanned receipts (Hugging Face)
    real_charts: int = 30            # ChartQA chart images with questions (Hugging Face)
    real_sba_loans: int = 25000      # SBA 7(a) loan book rows (Kaggle)


@dataclass(frozen=True)
class LLM:
    """Model routing. Free tiers change often: check the provider pages before a session.

    Model strings use LiteLLM's "provider/model" format. The primary is tried first,
    then each fallback in order. OFFLINE mode never calls a provider: it answers from
    the on-disk cache or a deterministic stub, so tests and CI run without keys.
    """
    primary: str = env("LLM_PRIMARY", "gemini/gemini-3.5-flash-lite")
    fallbacks: tuple[str, ...] = tuple(
        m for m in env("LLM_FALLBACKS", "groq/openai/gpt-oss-20b,ollama/llama3.2").split(",") if m)
    vision: str = env("LLM_VISION", "gemini/gemini-3.5-flash-lite")
    judge: str = env("LLM_JUDGE", "gemini/gemini-3.5-flash-lite")
    embedding_model: str = env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    offline: bool = env("LLM_OFFLINE", "0") == "1"
    cache_enabled: bool = env("LLM_CACHE", "1") == "1"
    temperature: float = 0.0
    max_retries: int = 3
    requests_per_minute: int = int(env("LLM_RPM", "10"))


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    sizes: Sizes = field(default_factory=Sizes)
    llm: LLM = field(default_factory=LLM)
    statement_end_month: str = "2026-09"  # last full month of bank statements
    regions: tuple[str, ...] = ("North", "South", "East", "West")
    bank_name: str = "Good Bank"

    @property
    def database_url(self) -> str:
        return env("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'creditlens.db'}")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(env("LANGFUSE_PUBLIC_KEY") and env("LANGFUSE_SECRET_KEY"))

    @property
    def neo4j(self) -> dict[str, str]:
        return {"uri": env("NEO4J_URI"), "user": env("NEO4J_USERNAME", "neo4j"), "password": env("NEO4J_PASSWORD")}


settings = Settings()
