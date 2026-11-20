# Client brief: CreditLens AI for Good Bank

*Hand this to every pod on Day 1 before the discovery role-play. The hidden constraint cards are in the trainer runbook, not here.*

## The story students will recognise

Meet Rohan. He runs a small packaging company. Orders are growing, and he needs a new packaging line before the festive season. He applies to Good Bank for a business loan.

Meet Priya. Priya is a credit analyst at Good Bank, and she is handling Rohan's file. Before she can recommend anything, she has to read twelve months of bank statements, three years of financial statements, and a forty-page credit policy. She has to work out whether the owner runs other companies that already owe Good Bank money. She has to raise a ticket to compliance for every rule the application breaks. Then she writes a memo, and an underwriter decides.

Priya does this for every file. Rohan waits weeks, and mistakes slip through.

## What CreditLens AI does

CreditLens AI does Priya's paperwork so she can focus on the decision:

- It reads the bank statements and financial statements, including the scanned ones.
- It checks the application against every rule in the credit policy manual.
- It maps related-party exposure through shared directors and registered addresses.
- It scores default risk with a model trained on a real public loan book.
- It drafts the credit memo in Good Bank's standard format, with every number traced to its source.
- It raises each policy exception to compliance as a Jira ticket, exactly once.

Priya reviews the draft, the underwriter decides, and both of them can see why the system said what it said. Rohan gets his answer faster, and Good Bank makes fewer mistakes.

**One line for students:** Rohan needs a loan fast, and Priya is drowning in paperwork. We are building the AI assistant that does Priya's paperwork, so she can focus on the decision.

## What Good Bank asked for

Good Bank lends to small and mid-sized businesses. Its credit team wants CreditLens AI to:

1. Extract financials from bank statements and financial statements, including scanned ones.
2. Check each application against the credit policy manual.
3. Map related-party exposure through shared directors and guarantors.
4. Score default risk with a machine-learning model.
5. Draft the credit memo for underwriter approval.
6. Raise policy exceptions to compliance as Jira tickets.

A human underwriter makes every final decision. No automated system may approve or decline an application on its own (policy section 1).

## Who you will meet in discovery

| Role | What they care about |
| --- | --- |
| Head of Credit | Turnaround time, consistency between analysts, fewer missed exceptions |
| Chief Risk Officer | Explainability of every recommendation, model fairness, audit trail |
| Chief Information Security Officer | Borrower data never leaves the bank unmasked, who can see which region, read-only access for auditors |

Pods interview all three. Each one holds constraints they will not volunteer unless asked the right question.

## Delivery approach

The engagement follows Build → Deploy → Evaluate → Maintain:

- **Build**: data foundation, extraction, retrieval, agents, integrations.
- **Deploy**: containers, CI with an evaluation gate, a free-tier cloud deployment.
- **Evaluate**: a golden set labelled by humans on Day 2, scored at every later step, with an LLM judge calibrated against those labels.
- **Maintain**: tracing of every call, drift monitoring, incident drills, retrain or rollback.

## Return on investment

Pods build the ROI case from the figures in this table only. Do not estimate figures that are not here.

| Input | Value |
| --- | --- |
| Manual review hours per application | **[TRAINER TO FILL]** |
| Loaded cost per analyst hour | **[TRAINER TO FILL]** |
| Applications per month | **[TRAINER TO FILL]** |
| Share of applications with at least one policy exception | **[TRAINER TO FILL]** (the synthetic data has a known share; the trainer may use it) |
| Target review hours per application with CreditLens AI | **[TRAINER TO FILL]** |

These are the gap flagged in the planning document: the brief needs real numbers chosen by the trainer before Day 1.

## Data you will work with

Everything about Good Bank, Rohan and the other applicants is synthetic and generated from one random seed. Four real public datasets are plugged in where realism matters and no borrower privacy is at stake. See `docs/data_sources.md` for what each one is, where it comes from, and its licence.
