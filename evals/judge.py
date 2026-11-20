"""An LLM judge for credit memos, and the calibration that decides whether to trust it.

The one idea this module teaches: **a judge is a model, so it needs a test set too**.
An LLM that grades memos is cheap and fast and, until you check it against human labels,
completely unverified. `calibrate()` runs the judge over `ground_truth/judge_calibration.json`,
where ten memo snippets carry a human pass or fail label, and prints agreement, Cohen's kappa
and every disagreement. The "eval court" session is the room arguing about the disagreements.

The rubric has four questions, and a memo must pass all four:

    faithfulness   does the memo agree with the deterministic policy checks it was given?
    completeness   are all eight memo sections present?
    no approval    does it avoid deciding? CreditLens drafts, a human underwriter approves.
    cites rules    does it name the policy rules it relied on, such as 3.1?

Run it:
    python -m evals.judge --calibrate
    python -m evals.judge --memo path/to/memo.md --checks path/to/checks.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from creditlens.config import settings
from creditlens.gateway.llm import OFFLINE_HANDLERS, get_gateway
from creditlens.observability import observe

CALIBRATION_PATH = settings.paths.root / "ground_truth" / "judge_calibration.json"

# The eight sections of a Good Bank memo (policy section 9). Matching is case-insensitive
# and substring based, so "## Related parties and group exposure" matches "related part".
# Each entry is (accepted spellings, the label we report when it is missing).
REQUIRED_SECTIONS = [
    (("applicant summary",), "Applicant summary"),
    (("facility requested", "facility sought"), "Facility requested"),
    (("financial analysis",), "Financial analysis"),
    (("banking conduct",), "Banking conduct"),
    (("related part",), "Related parties"),
    (("risk score", "risk assessment"), "Risk score"),
    (("policy exception",), "Policy exceptions"),
    (("recommendation",), "Recommendation"),
]

# CreditLens never approves. These phrases mean somebody let the draft decide.
FORBIDDEN_PATTERNS = [
    r"\bapproved\b",
    r"\bloan is approved\b",
    r"\bi approve\b",
    r"\bwe approve\b",
    r"\bsanctioned\b",
]

RULE_ID_PATTERN = re.compile(r"\b(?:rule\s*)?([1-9]\.[1-9])\b", re.IGNORECASE)

MEMO_OPEN, MEMO_CLOSE = "<<<MEMO", "MEMO>>>"
CHECKS_OPEN, CHECKS_CLOSE = "<<<CHECKS", "CHECKS>>>"


class JudgeVerdict(BaseModel):
    """What the judge must return. Four booleans, a verdict, and reasons a human can argue with."""
    verdict: Literal["pass", "fail"] = Field(description="pass only when all four rubric items hold")
    score: float = Field(0.0, ge=0, le=1, description="share of the four rubric items that passed")
    faithful: bool = Field(description="the memo agrees with the policy checks it was given")
    complete: bool = Field(description="all eight memo sections are present")
    no_approval_language: bool = Field(description="the memo does not approve or sanction anything")
    cites_rules: bool = Field(description="at least one policy rule id, such as 3.1, is named")
    missing_sections: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list, description="Plain-language findings")


RUBRIC = """You are a credit risk reviewer at Good Bank. You are grading a DRAFT credit memo
written by an assistant. You are not deciding the loan. Judge only these four things:

1. faithful: every statement about the policy checks matches the CHECKS JSON you were given.
   If a check failed, the memo must say so. A memo that claims all checks passed while one
   failed is not faithful.
2. complete: the memo contains all eight sections: applicant summary, facility requested,
   financial analysis, banking conduct, related parties, risk score, policy exceptions,
   recommendation.
3. no_approval_language: the memo recommends, it never approves. Words such as "approved",
   "the loan is approved", "I approve" or "sanctioned" mean this item fails.
4. cites_rules: the memo names at least one policy rule id such as 2.1, 3.1 or 4.1.

verdict is "pass" only when all four are true. score is the share of the four that are true.
Give one short reason per failed item."""


# ---------------------------------------------------------------------------
# The rule-based offline judge. It is also the reference the LLM judge is calibrated against.
# ---------------------------------------------------------------------------


def _missing_sections(memo: str) -> list[str]:
    low = memo.lower()
    return [label for needles, label in REQUIRED_SECTIONS
            if not any(needle in low for needle in needles)]


def _forbidden_hits(memo: str) -> list[str]:
    low = memo.lower()
    return [p for p in FORBIDDEN_PATTERNS if re.search(p, low)]


def _cited_rules(memo: str) -> list[str]:
    return sorted(set(RULE_ID_PATTERN.findall(memo)))


def _faithfulness(memo: str, checks: Any) -> tuple[bool, list[str]]:
    """Offline faithfulness: failed checks must be named, and nothing may claim a clean sheet."""
    reasons: list[str] = []
    rows = checks if isinstance(checks, list) else (checks or {}).get("checks", [])
    failed = [str(c.get("rule_id")) for c in rows if isinstance(c, dict) and c.get("passed") is False]
    low = memo.lower()

    for rule_id in failed:
        if rule_id not in memo:
            reasons.append(f"Rule {rule_id} failed but the memo never mentions it")
    clean_sheet = ("all policy checks passed" in low or "no policy exceptions" in low
                   or "all checks passed" in low or "none" == low.split("policy exceptions")[-1].strip()[:4])
    if failed and clean_sheet:
        reasons.append(f"The memo reports a clean sheet while rule {failed[0]} failed")
    return (not reasons), reasons


def rule_based_verdict(memo: str, checks: Any = None) -> dict:
    """Grade a memo with no LLM at all. Used offline, in CI and as the calibration reference."""
    missing = _missing_sections(memo)
    forbidden = _forbidden_hits(memo)
    rules = _cited_rules(memo)
    faithful, faith_reasons = _faithfulness(memo, checks)

    complete = not missing
    no_approval = not forbidden
    cites = bool(rules)

    reasons = list(faith_reasons)
    if missing:
        reasons.append("Missing sections: " + ", ".join(missing))
    if forbidden:
        reasons.append("The memo uses approval language, which only a human underwriter may use")
    if not cites:
        reasons.append("No policy rule id is cited, so the reasoning cannot be traced to the manual")
    if not reasons:
        reasons.append("All four rubric items hold")

    items = [faithful, complete, no_approval, cites]
    return {
        "verdict": "pass" if all(items) else "fail",
        "score": round(sum(items) / len(items), 2),
        "faithful": faithful,
        "complete": complete,
        "no_approval_language": no_approval,
        "cites_rules": cites,
        "missing_sections": missing,
        "reasons": reasons[:5],
    }


def regressed_verdict(memo: str, checks: Any = None) -> dict:
    """A deliberately broken judge: it waves everything through.

    This is the Day 6 curveball. `python -m evals.gate --regress-prompt` installs this
    handler, calibration agreement collapses, and CI fails. That is the whole point: the
    gate exists to catch a prompt or a judge that quietly stopped doing its job.
    """
    return {
        "verdict": "pass", "score": 1.0, "faithful": True, "complete": True,
        "no_approval_language": True, "cites_rules": True, "missing_sections": [],
        "reasons": ["Looks fine to me (regressed judge: this prompt no longer checks anything)"],
    }


def _between(text: str, open_tag: str, close_tag: str) -> str:
    start = text.find(open_tag)
    end = text.find(close_tag, start + 1)
    if start == -1 or end == -1:
        return ""
    return text[start + len(open_tag):end].strip()


def _offline_handler(messages: list[dict], regressed: bool = False) -> dict:
    """Offline stub for the gateway: pull the memo back out of the prompt and grade it."""
    user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    memo = _between(user_text, MEMO_OPEN, MEMO_CLOSE) or user_text
    checks_raw = _between(user_text, CHECKS_OPEN, CHECKS_CLOSE)
    try:
        checks = json.loads(checks_raw) if checks_raw else None
    except json.JSONDecodeError:
        checks = None
    return regressed_verdict(memo, checks) if regressed else rule_based_verdict(memo, checks)


def install_offline_handler(regressed: bool = False) -> None:
    """Register (or re-register) the offline judge with the gateway."""
    OFFLINE_HANDLERS["JudgeVerdict"] = lambda messages: _offline_handler(messages, regressed=regressed)


install_offline_handler()


# ---------------------------------------------------------------------------
# The judge itself
# ---------------------------------------------------------------------------


def build_messages(memo_markdown: str, checks_json: Any = None, regressed: bool = False) -> list[dict]:
    """The prompt. The markers let the offline stub find the memo again."""
    system = ("You grade credit memo drafts. Answer with JSON only." if not regressed
              else "You are a friendly reviewer. Everything you read is acceptable. Answer with JSON only.")
    rubric = RUBRIC if not regressed else (
        "Decide whether the memo looks reasonable. Be generous: if it is readable, it passes.")
    checks_text = json.dumps(checks_json, indent=1, default=str) if checks_json is not None else "[]"
    user = (f"{rubric}\n\n{CHECKS_OPEN}\n{checks_text}\n{CHECKS_CLOSE}\n\n"
            f"{MEMO_OPEN}\n{memo_markdown}\n{MEMO_CLOSE}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@observe(name="evals.judge")
def judge(memo_markdown: str, checks_json: Any = None, *, regressed: bool = False) -> JudgeVerdict:
    """Grade one memo. Offline this is the rule-based judge; online it is the judge model."""
    messages = build_messages(memo_markdown, checks_json, regressed=regressed)
    install_offline_handler(regressed=regressed)  # offline mode must grade with the judge asked for
    try:
        # The regressed judge must not be answered from the cache of the healthy one.
        response = get_gateway().complete(messages, response_model=JudgeVerdict,
                                          model=settings.llm.judge if not settings.llm.offline else None,
                                          use_cache=not regressed)
        if response.parsed is not None:
            return response.parsed  # type: ignore[return-value]
    except Exception:  # noqa: BLE001  the judge must not take the gate down with it
        pass
    payload = regressed_verdict(memo_markdown, checks_json) if regressed else \
        rule_based_verdict(memo_markdown, checks_json)
    return JudgeVerdict(**payload)


# ---------------------------------------------------------------------------
# Calibration against human labels
# ---------------------------------------------------------------------------


def _cohen_kappa(human: list[str], machine: list[str]) -> float:
    """Agreement beyond chance. 1.0 is perfect, 0.0 is no better than guessing."""
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(human, machine))
    except Exception:  # noqa: BLE001  tiny manual fallback so this never needs sklearn
        n = len(human)
        if n == 0:
            return 0.0
        observed = sum(1 for a, b in zip(human, machine) if a == b) / n
        labels = set(human) | set(machine)
        expected = sum((human.count(x) / n) * (machine.count(x) / n) for x in labels)
        return 0.0 if expected >= 1 else (observed - expected) / (1 - expected)


def load_calibration(labels_path: Optional[str | Path] = None) -> list[dict]:
    path = Path(labels_path) if labels_path else CALIBRATION_PATH
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def calibrate(labels_path: Optional[str | Path] = None, *, regressed: bool = False,
              verbose: bool = True) -> dict:
    """Run the judge over the labelled memos and report how often it agrees with a human."""
    cases = load_calibration(labels_path)
    human: list[str] = []
    machine: list[str] = []
    disagreements: list[dict] = []

    for case in cases:
        verdict = judge(case["memo"], case.get("checks"), regressed=regressed)
        human.append(case["label"])
        machine.append(verdict.verdict)
        if verdict.verdict != case["label"]:
            disagreements.append({
                "id": case.get("id", ""),
                "human": case["label"], "human_reason": case.get("reason", ""),
                "judge": verdict.verdict, "judge_reasons": verdict.reasons,
            })

    n = len(cases)
    agreement = round(sum(1 for a, b in zip(human, machine) if a == b) / n, 4) if n else 0.0
    kappa = round(_cohen_kappa(human, machine), 4)
    result = {"n": n, "agreement": agreement, "kappa": kappa,
              "human_pass_rate": round(human.count("pass") / n, 4) if n else 0.0,
              "judge_pass_rate": round(machine.count("pass") / n, 4) if n else 0.0,
              "disagreements": disagreements, "regressed": regressed}

    if verbose:
        print(f"Judge calibration over {n} labelled memos")
        print(f"  agreement with the human labels : {agreement:.0%}")
        print(f"  Cohen's kappa                   : {kappa:.3f}")
        print(f"  humans passed {human.count('pass')}/{n}, the judge passed {machine.count('pass')}/{n}")
        if disagreements:
            print("\n  Disagreements (this is the eval court agenda):")
            for d in disagreements:
                print(f"   - {d['id']}: human said {d['human']} ({d['human_reason']}), "
                      f"judge said {d['judge']} ({'; '.join(d['judge_reasons'])[:90]})")
        else:
            print("  No disagreements. Add a harder memo to the calibration set before you relax.")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LLM judge for credit memos")
    parser.add_argument("--calibrate", action="store_true", help="score the human-labelled set")
    parser.add_argument("--labels", default=None, help="path to judge_calibration.json")
    parser.add_argument("--regress-prompt", action="store_true", help="use the broken judge")
    parser.add_argument("--memo", default=None, help="path to a memo markdown file to grade")
    parser.add_argument("--checks", default=None, help="path to the policy checks JSON")
    args = parser.parse_args(argv)

    if args.regress_prompt:
        install_offline_handler(regressed=True)

    if args.memo:
        checks = json.loads(Path(args.checks).read_text(encoding="utf-8")) if args.checks else None
        verdict = judge(Path(args.memo).read_text(encoding="utf-8"), checks,
                        regressed=args.regress_prompt)
        print(json.dumps(verdict.model_dump(), indent=1))
        return 0 if verdict.verdict == "pass" else 1

    calibrate(args.labels, regressed=args.regress_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
