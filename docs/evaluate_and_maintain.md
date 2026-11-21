# Evaluate and Maintain (Day 7)

*Beginner notes for milestone 11. Code: `evals/`, `monitoring/`, `src/creditlens/gateway/`.*

## Offline evaluation versus online observability

| | Offline evaluation | Online observability |
| --- | --- | --- |
| When | before a change ships | all the time |
| Against | the golden set, human labels | traces, metrics, drift |
| Tools | `evals/` and the CI gate | Langfuse, `monitoring/` |
| Question | is this version better? | is production still healthy? |

You need both. The gate stops a bad change; the traces tell you the world changed.

## The CI gate

`python -m evals.gate` runs extraction accuracy, retrieval recall and judge calibration in offline
mode, compares each with `evals/thresholds.json`, and exits non-zero if any metric fell. The GitHub
Actions workflow runs it after the tests. `--regress-prompt` swaps in a deliberately broken judge so
the room can watch the gate fail. Do not lower a threshold to make the build green: look at the change
that moved the metric.

## The LLM judge and the eval court

A judge model grades memos against a rubric: faithful to the checks, all eight sections present, no
approval language, cites rule ids. A judge is only as good as its agreement with humans, so
`ground_truth/judge_calibration.json` holds memo snippets with human pass/fail labels and reasons.

```bash
python -m evals.judge --calibrate
```

Agreement and Cohen's kappa are printed with every disagreement. The eval court: one pod argues for
the human, one for the judge, the room decides whether to change the rubric or the label. One case is
deliberately borderline (a memo that ticks every box but calls a breached file "clean"), because a rule
cannot see tone and a human can.

## Trajectory checks

A memo can read well and still be wrong about how it was made: the policy checker never ran, Jira
was called twice, a forbidden tool was used. `evals/trajectory.py` takes the recorded tool calls from a
run and checks required tools, forbidden tools, per-tool budgets, ordering (checks before the memo
writer) and identical repeated calls.

## Gateway resilience

Free tiers are small and change often. `creditlens.gateway.llm` is the one door to every model:

- **Cache**: same prompt, same model, same answer, zero quota. Ship `data/cache/llm/` to the room.
- **Rate limit**: `LLM_RPM` requests per minute per process.
- **Fallbacks**: primary, then each fallback in order (`LLM_PRIMARY`, `LLM_FALLBACKS`), then Ollama.
- **Offline mode**: `LLM_OFFLINE=1` answers from the cache or a deterministic stub.

`src/creditlens/gateway/litellm_config.yaml` is the same routing for a whole team behind one LiteLLM
proxy URL. The drill:

```bash
python -m creditlens.gateway.drill -n 6                  # who answered, cache hits, latency
python -m creditlens.gateway.drill -n 3 --kill-primary   # the trainer cuts the primary live
```

`costs.py` holds no prices on purpose. Set `LLM_PRICES_JSON` to your provider's current table and the
drill estimates spend; otherwise it says it cannot, which is the honest answer.

## Drift

`monitoring/drift.py` compares the SBA training features (reference) with the features the adapter
produces for today's applications (current), with Evidently when it imports and a per-column KS test
otherwise. `--inject-drift` inflates amounts and shifts terms so the report has something to flag.
A drifted column is not automatically a problem: the approval year always drifts because the training
data stops in 2014. Amount and term drifting together is the signal worth paging someone about.

## The four incident drills

```bash
python -m monitoring.incidents --scenario data_drift
python -m monitoring.incidents --scenario cost_spike
python -m monitoring.incidents --scenario policy_update
python -m monitoring.incidents --scenario prompt_injection
```

Each prints a three-line runbook: the symptom, where to look, the fix.

| Incident | Symptom | Where to look | Fix |
| --- | --- | --- | --- |
| Data drift | risk bands shift for a whole batch | drift summary, Evidently report | retrain, compare, move the alias, or roll back |
| Cost spike | requests per minute climb, memos do not | Langfuse cost per trace by user | cache on, lower RPM, per-user budget in the proxy, alert on tokens per day |
| Policy update | memo cites the old threshold, golden questions fail | retrieval eval, the cited chunk id in the trace | re-index, update RULES and the answer key together, bump the prompt version, re-run the gate |
| Prompt injection | a recommendation flips, or the output rail logs approval language | guardrail log, the memo's trace | both rails on; the memo writer sees aggregated features, never raw narrations |

## Retrain versus rollback

Retrain when the world changed (drift) and you have new labels. Roll back when the change was yours
(a worse model, a regressed prompt) and the previous version is known good. Both are one step: move the
MLflow alias, or pin the prompt version. Rehearse both before handoff.
