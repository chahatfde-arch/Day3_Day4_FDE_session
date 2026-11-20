"""Milestone 9: the three guardrails that stand between a borrower's file and an LLM.

1. `pii` strips director PAN, phone, email, address and names before any prompt leaves
   the building. Presidio does the work when it is installed; a plain-regex fallback
   does it when it is not, so CI and a laptop without a spaCy model behave the same.
2. `check` refuses prompts that try to steer the model: "ignore previous instructions",
   "approve this loan", "what is the director's PAN". NeMo Guardrails when available,
   regex otherwise.
3. `rbac` turns a `User` into a SQL WHERE fragment and a vector-store metadata filter,
   so an analyst's question can only ever touch their own region.

Nothing here is a substitute for the human underwriter. Guardrails reduce the number
of bad prompts and bad answers; they do not make the system safe on their own.
"""
from __future__ import annotations

__all__ = ["pii", "check", "rbac"]
