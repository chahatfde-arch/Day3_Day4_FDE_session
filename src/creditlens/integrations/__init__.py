"""Milestone 7: the systems a real bank already has, and how a new agent reaches them.

Three of them, each with a different lesson:

* `oracle.py`      a core banking database with 1990s column names -> an **adapter**
* `soap.py`        a slow, flaky SOAP service -> **timeouts, retries, a circuit breaker**
* `slack.py`       where the underwriter actually is -> **approve from a button**

Every one of them runs without the real system: a SQLite file stands in for Oracle, a
local spyne service (or a deterministic generator) stands in for the SOAP endpoint, and the
Slack app has a `--dry-run` that just prints the message.
"""
