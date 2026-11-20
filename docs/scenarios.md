# Story scenarios: Rohan and Rahul

Two extra applications, woven through Sessions 1 and 2. They are **opt-in**: they are written to
`data/scenarios/`, never to `data/raw/`, and they use their own random stream, so `generate_all`
and every exact number already in the workshop stay the same.

```bash
python -m creditlens.scenarios      # writes data/scenarios and prints both case files
python -m pytest tests/test_scenarios.py -q
```

All people and companies are invented.

## The two cases

| | Rohan (A901, company SC01) | Rahul (A902, company SC10) |
|---|---|---|
| First look | Clean file, steady inflows, asks INR 5.0 crore | Thin statement, tight cash, asks INR 3.0 crore |
| Session 1 reveal | 24 round trips to RAVI CONTAINERS (three spellings, one name), 2.07 crore out and back, 40% of inflow from that one name | 36 cash deposits, each followed next day by the same amount to KAMLA DEVI; statement shows 60% of declared turnover |
| Session 2 reveal | Director link to SC02 and SC03, address link SC03 to SC04: a group of four, exposure 155,000 against the 120,000 limit. Alone: 50,000, passes | Same address as Kamla Fabrics, whose director is Kamla Devi (his mother). Group exposure 55,000, within the limit |
| Lesson | Only the graph shows the breach of rule 5.2 | The system explains why he looks weak. It does not accuse him |

## Where each reveal lives in the code

- `generate.py`: builds the registry, applications and two statements in the pipeline's own layouts,
  plus `answer_key/scenarios.json` (instructor only).
- `detect.py`: `round_trips`, `top_counterparties`, `related_flows`, `pass_through`, `revenue_gap`, and
  `case_file`, which combines the cleaned statement with `GraphStore.group_exposure`.
- Session 1 uses the real cleaning step (`clean_statement`): normalising the names is what makes the
  round trips visible. Session 2 uses the real graph (`GraphStore`), the same code as the hidden-group hunt.

## The rule that keeps this honest

A flag is a lead for a person, not a verdict. `case_file` returns flags with evidence and the note
"Leads for a human to check. Not a finding, and not a decision." Nothing in the pack uses the word fraud,
and the bank's policy still forbids automated approvals. Whether keeping money in a family account is a
tax matter is for the right team to judge, not for this assistant.

## Safe to change

The thresholds (`window_days`, `tolerance`) are arguments. With a one percent tolerance only 3 of Rohan's
24 round trips survive, which is the point of the Session 1 exercise: a knob that hides the pattern is a
bad default, so someone has to choose it and say why.

## Four bonus cases (deep dive only)

Appended to the same pack on their own random stream, so Rohan's and Rahul's files are byte-for-byte unchanged.

| Case | What it teaches | Flags |
|---|---|---|
| A903 Meena, honest baker | Rule 4.1 fires, and the season explains it | `RULE_4_1`, `SEASONAL_PATTERN` |
| A904 Tapti Trading | Window dressing: 77% of the year's inflow in the last 30 days | `WINDOW_DRESSING` |
| A905 Sagar Wellness | Declared healthcare, 73% of inflow from betting-related names (rule 2.2) | `SECTOR_MISMATCH` |
| A906 Konark Exports | FY2026 EBITDA does not equal revenue minus costs; the statement shows 55% of declared revenue | `FOOTING`, `REVENUE_GAP` |

## Run it on Google Colab

`notebooks/08_story_scenarios.ipynb` runs all six cases, one section at a time, with charts. It clones the public
pre-work repo, writes the scenario code into it, and needs no API key. Upload the notebook to Colab
(File, Upload notebook) and run the setup section first.
