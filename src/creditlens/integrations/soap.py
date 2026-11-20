"""The legacy SOAP service, and how to call something that is slow and sometimes lies.

Banks keep SOAP services for decades because rewriting them is expensive and risky. Your
agent will meet one. The three things that keep it from taking your whole system down with
it are in this file:

* **Timeout.** Never wait forever. A request that has not answered in five seconds is not
  going to.
* **Retry with backoff.** A transient failure deserves a second try, after a pause that
  doubles each time, so a struggling service is not hammered by its own clients.
* **Circuit breaker.** After enough failures in a row, stop calling at all for a while.
  `closed` means traffic flows, `open` means every call fails instantly without touching
  the network, `half_open` means "let one request through and see". Without this, every
  thread in your service ends up queued behind a dead dependency.

When no SOAP endpoint is reachable the client returns the same shape from a local
deterministic generator, flagged `source="mock"`, so the agent keeps working and the memo
says plainly where the numbers came from.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import date, timedelta
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from creditlens.observability import observe

DEFAULT_WSDL = os.environ.get("SOAP_WSDL_URL", "http://localhost:8001/?wsdl")


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------


class RepaymentEntry(BaseModel):
    due_date: str = Field(description="When the instalment was due, ISO date")
    amount: float = Field(description="Instalment amount in INR thousands")
    paid_on: Optional[str] = Field(None, description="When it was actually paid, or None")
    days_late: int = 0


class RepaymentHistory(BaseModel):
    customer_id: str
    entries: list[RepaymentEntry] = Field(default_factory=list)
    source: Literal["soap", "mock"] = "soap"
    endpoint: str = ""

    @property
    def late_count(self) -> int:
        return sum(1 for e in self.entries if e.days_late > 0)

    @property
    def worst_delay_days(self) -> int:
        return max((e.days_late for e in self.entries), default=0)


# ---------------------------------------------------------------------------
# The deterministic generator. The server and the offline fallback both use it,
# so the numbers are identical whether or not a service is running.
# ---------------------------------------------------------------------------


def _seed_for(customer_id: str) -> int:
    return int(hashlib.sha256(customer_id.encode()).hexdigest()[:8], 16)


def generate_history(customer_id: str, months: int = 12,
                     start: date = date(2025, 10, 5)) -> list[dict]:
    """Twelve instalments, always the same for the same customer id."""
    seed = _seed_for(customer_id)
    amount = 100 + (seed % 900)
    rows = []
    for i in range(months):
        due = start + timedelta(days=30 * i)
        # A cheap, reproducible "was this one late" rule. No randomness at call time.
        bucket = (seed >> (i % 16)) % 10
        days_late = 0 if bucket < 7 else (bucket - 6) * 4
        paid = due + timedelta(days=days_late)
        rows.append({"due_date": due.isoformat(), "amount": float(amount),
                     "paid_on": paid.isoformat(), "days_late": int(days_late)})
    return rows


# ---------------------------------------------------------------------------
# The circuit breaker
# ---------------------------------------------------------------------------


class CircuitOpen(RuntimeError):
    """Raised instead of making a call, because the breaker is open."""


class CircuitBreaker:
    """closed -> (too many failures) -> open -> (after reset_after) -> half_open -> closed.

    `clock` is injectable so a unit test can prove the timing without sleeping.
    """

    def __init__(self, failures: int = 3, reset_after: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failures
        self.reset_after = reset_after
        self._clock = clock
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0

    @property
    def failure_count(self) -> int:
        return self._failures

    @property
    def state(self) -> str:
        """closed, open or half_open. Reading it is what lets an open breaker time out."""
        if self._state == "open" and (self._clock() - self._opened_at) >= self.reset_after:
            return "half_open"
        return self._state

    def reset(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self._clock()

    def record_failure(self) -> None:
        self._failures += 1
        if self.state == "half_open" or self._failures >= self.failure_threshold:
            self._trip()

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run `fn` unless the breaker is open, and update the state from the result."""
        current = self.state
        if current == "open":
            raise CircuitOpen(f"circuit is open; retry in "
                              f"{self.reset_after - (self._clock() - self._opened_at):.1f}s")
        was_half_open = current == "half_open"
        if was_half_open:
            self._state = "half_open"
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class RepaymentClient:
    """Fetch a customer's repayment history from the legacy SOAP service."""

    def __init__(self, wsdl: Optional[str] = None, timeout: float = 5.0, retries: int = 3,
                 backoff: float = 0.25, breaker: Optional[CircuitBreaker] = None,
                 allow_mock: bool = True):
        self.wsdl = wsdl or DEFAULT_WSDL
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff = backoff
        self.breaker = breaker or CircuitBreaker(failures=3, reset_after=30)
        self.allow_mock = allow_mock

    # --- the zeep call ------------------------------------------------------------
    def _soap_call(self, customer_id: str) -> list[dict]:
        from zeep import Client
        from zeep.transports import Transport

        transport = Transport(timeout=self.timeout, operation_timeout=self.timeout)
        client = Client(self.wsdl, transport=transport)
        rows = client.service.get_history(customer_id)
        return [{"due_date": str(getattr(r, "due_date", "")),
                 "amount": float(getattr(r, "amount", 0) or 0),
                 "paid_on": (str(r.paid_on) if getattr(r, "paid_on", None) else None),
                 "days_late": int(getattr(r, "days_late", 0) or 0)}
                for r in (rows or [])]

    # --- the public call ----------------------------------------------------------
    @observe(name="integrations.soap.get_history")
    def get_history(self, customer_id: str) -> RepaymentHistory:
        """Try the service, with retries and a breaker. Fall back to the local generator."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                rows = self.breaker.call(self._soap_call, customer_id)
                return RepaymentHistory(customer_id=customer_id,
                                        entries=[RepaymentEntry(**r) for r in rows],
                                        source="soap", endpoint=self.wsdl)
            except CircuitOpen as e:
                last_error = e
                break  # no point retrying while the breaker is open
            except Exception as e:  # noqa: BLE001  network, SOAP fault, bad WSDL, missing zeep
                last_error = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))  # 0.25s, 0.5s, 1s, ...
        if not self.allow_mock:
            raise RuntimeError(f"repayment service unavailable: {last_error}")
        return self.mock_history(customer_id)

    def mock_history(self, customer_id: str) -> RepaymentHistory:
        """The same shape, generated locally, clearly labelled."""
        return RepaymentHistory(customer_id=customer_id,
                                entries=[RepaymentEntry(**r) for r in generate_history(customer_id)],
                                source="mock", endpoint=self.wsdl)


def get_repayment_history(company_id: str, **kwargs: Any) -> RepaymentHistory:
    """Convenience wrapper used by the agent tool."""
    return RepaymentClient(**kwargs).get_history(company_id)


def main() -> int:  # pragma: no cover - a hand demo
    import argparse

    parser = argparse.ArgumentParser(prog="python -m creditlens.integrations.soap")
    parser.add_argument("customer_id")
    parser.add_argument("--wsdl", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    history = RepaymentClient(wsdl=args.wsdl, timeout=args.timeout).get_history(args.customer_id)
    print(f"{history.customer_id}: {len(history.entries)} instalments from {history.source}, "
          f"{history.late_count} late, worst delay {history.worst_delay_days} days")
    for entry in history.entries:
        print(f"  due {entry.due_date}  amount {entry.amount}  paid {entry.paid_on}  "
              f"late {entry.days_late} days")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
