"""The core banking system: Oracle in production, SQLite on a laptop.

The one idea this module teaches: **an adapter is a translation layer, and the translation
is data, not code.** The core banking table was designed in 1998 and its columns are called
CUST_ID, OUTSTANDING_AMT and PRODUCT_CD. CreditLens calls them company_id,
outstanding_inr_000 and product. Put that mapping in a dictionary and the rest of the
system never has to know the old names:

    COLUMN_MAP = {"CUST_ID": "company_id", "OUTSTANDING_AMT": "outstanding_inr_000",
                  "PRODUCT_CD": "product"}

When ORACLE_DSN is set the adapter uses python-oracledb in thin mode (no Oracle client to
install). Otherwise it builds a SQLite copy of CORE_LOANS from `existing_loans.csv`, with
the same uppercase column names, so the exact same SQL and the exact same mapping run.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from creditlens.agents.adapters import cache_dir, raw_dir
from creditlens.observability import observe

#: Oracle column -> CreditLens field. The whole translation lives here.
COLUMN_MAP: dict[str, str] = {
    "CUST_ID": "company_id",
    "OUTSTANDING_AMT": "outstanding_inr_000",
    "PRODUCT_CD": "product",
}

TABLE = "CORE_LOANS"
SELECT_SQL = f"SELECT {', '.join(COLUMN_MAP)} FROM {TABLE} WHERE CUST_ID = "

#: What the mock core banking system calls each Good Bank product.
DEFAULT_PRODUCT_CODE = "WCDL"  # working capital demand loan


def sqlite_fallback_path() -> Path:
    override = os.environ.get("CREDITLENS_CORE_BANKING_DB")
    return Path(override) if override else cache_dir() / "core_banking.db"


class CoreBankingAdapter:
    """Read a company's outstanding exposure from the core banking system."""

    def __init__(self, dsn: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, sqlite_path: Optional[Path] = None):
        self.dsn = dsn if dsn is not None else os.environ.get("ORACLE_DSN", "")
        self.user = user if user is not None else os.environ.get("ORACLE_USER", "creditlens")
        self.password = password if password is not None else os.environ.get("ORACLE_PASSWORD", "")
        self.sqlite_path = Path(sqlite_path) if sqlite_path else sqlite_fallback_path()

    # --- which backend ------------------------------------------------------------
    @property
    def backend(self) -> str:
        return "oracle" if self.dsn else "sqlite-fallback"

    # --- the public call ----------------------------------------------------------
    @observe(name="integrations.core_banking.get_exposure")
    def get_exposure(self, company_id: str) -> dict:
        """Return `{company_id, outstanding_inr_000, product, source}`.

        A company the core banking system has never lent to is not an error: it has an
        outstanding balance of zero.
        """
        source = self.backend
        try:
            rows = self._fetch(company_id)
        except Exception:  # noqa: BLE001  Oracle configured but unreachable: use the stand-in, and say so
            if not self.dsn:
                raise
            rows, source = self._fetch_sqlite(company_id), "sqlite-fallback"
        mapped = [self._map_row(row) for row in rows]
        total = sum(int(row.get("outstanding_inr_000") or 0) for row in mapped)
        return {
            "company_id": company_id,
            "outstanding_inr_000": total,
            "product": mapped[0]["product"] if mapped else None,
            "facility_count": len(mapped),
            "source": source,
            "table": TABLE,
        }

    def _map_row(self, row: tuple) -> dict:
        """Apply COLUMN_MAP. The order of COLUMN_MAP is the order of the SELECT."""
        out = {}
        for (oracle_column, field), value in zip(COLUMN_MAP.items(), row):
            if field == "outstanding_inr_000":
                value = int(value or 0)
            out[field] = value
        return out

    # --- the two backends ---------------------------------------------------------
    def _fetch(self, company_id: str) -> list[tuple]:
        if self.dsn:
            return self._fetch_oracle(company_id)
        return self._fetch_sqlite(company_id)

    def _fetch_oracle(self, company_id: str) -> list[tuple]:  # pragma: no cover - needs Oracle
        import oracledb  # thin mode: no Oracle Instant Client needed

        # A connect timeout, so a down database costs seconds, not the default wait.
        with oracledb.connect(user=self.user, password=self.password, dsn=self.dsn,
                              tcp_connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute(SELECT_SQL + ":cust", cust=company_id)
                return cursor.fetchall()

    def _fetch_sqlite(self, company_id: str) -> list[tuple]:
        path = self.ensure_sqlite()
        with sqlite3.connect(path) as connection:
            cursor = connection.execute(SELECT_SQL + "?", (company_id,))
            return cursor.fetchall()

    # --- building the stand-in ----------------------------------------------------
    def ensure_sqlite(self, rebuild: bool = False) -> Path:
        """Create the CORE_LOANS stand-in from `existing_loans.csv` if it is not there yet."""
        path = self.sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not rebuild:
            return path
        rows = self._registry_rows()
        with sqlite3.connect(path) as connection:
            connection.execute(f"DROP TABLE IF EXISTS {TABLE}")
            connection.execute(
                f"CREATE TABLE {TABLE} (CUST_ID TEXT NOT NULL, OUTSTANDING_AMT INTEGER NOT NULL, "
                "PRODUCT_CD TEXT NOT NULL, OPENED_ON TEXT)")
            connection.executemany(
                f"INSERT INTO {TABLE} (CUST_ID, OUTSTANDING_AMT, PRODUCT_CD, OPENED_ON) "
                "VALUES (?, ?, ?, ?)", rows)
            connection.commit()
        return path

    @staticmethod
    def _registry_rows() -> list[tuple]:
        source = raw_dir() / "registry" / "existing_loans.csv"
        if not source.exists():
            return []
        rows = []
        with source.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append((row["company_id"], int(row["outstanding_inr_000"]),
                             DEFAULT_PRODUCT_CODE, "2024-04-01"))
        return rows


def get_core_banking_exposure(company_id: str, **kwargs: Any) -> dict:
    """Convenience wrapper used by the agent tool."""
    return CoreBankingAdapter(**kwargs).get_exposure(company_id)


def main() -> int:  # pragma: no cover - a hand demo
    import argparse

    parser = argparse.ArgumentParser(prog="python -m creditlens.integrations.oracle")
    parser.add_argument("company_id")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the SQLite stand-in")
    args = parser.parse_args()
    adapter = CoreBankingAdapter()
    if args.rebuild:
        adapter.ensure_sqlite(rebuild=True)
    print(adapter.get_exposure(args.company_id))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
