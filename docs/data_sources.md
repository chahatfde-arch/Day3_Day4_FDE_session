# Data sources and licences

CreditLens AI mixes one synthetic story with four real public datasets. Session 1 needs only the synthetic data and the receipts; Session 2 adds the other three. The rule is simple: anything that
describes a borrower (names, PAN, phones, addresses, bank transactions, financial statements of the
applicants) is synthetic, because real borrower data must never enter a training project. Real data is
used where it makes the engineering realistic and carries no Good Bank borrower privacy.

| Session | Dataset | What it is | Source | Licence | Sample size in the repo |
| --- | --- | --- | --- | --- | --- |
| Session 1 (notebooks 00, 01, 02) | Good Bank registry, applications, bank statements, financial statements, policy manual | Generated from one seed by `python -m creditlens.generators` | This repo | Project licence | 40 companies, 30 applications, 30 statements, 30 financial PDFs |
| Session 1 (notebook 02) | ICDAR 2019 SROIE scanned receipts | Real photographed receipts with human labels for company, date, address and total | Hugging Face `jsdnrs/ICDAR2019-SROIE` | CC BY 4.0 | 40 receipts (about 7 MB) |
| Session 2 (notebook 03) | SBA SOP 50 10 | The US Small Business Administration's lender rulebook for 7(a) and 504 loans, as Markdown | sba.gov (US government work, public domain) | Public domain | one document, about 146,000 words |
| Session 2 (notebook 03) | ChartQA | Real chart images with human-written questions and answers | Hugging Face `HuggingFaceM4/ChartQA` | GPL 3.0 (dataset) | 30 charts (about 1 MB) |
| Session 2 (notebooks 06, 07) | SBA 7(a) loan book, "Should This Loan be Approved or Denied?" | Real small-business loans 1987 to 2014 with a charged-off flag | Kaggle `mirbektoktogaraev/should-this-loan-be-approved-or-denied` | CC BY-SA 4.0 | 25,000 loans sampled from about 900,000 (2 MB) |

Fetch all of them with:

```bash
python -m creditlens.data_real            # everything, about a minute on a normal connection
python -m creditlens.data_real --only sba # one dataset
```

The samples land in `data/real/`, which is ignored by Git. Nothing real is committed to the repo.

## Why these four

- **SROIE receipts** are what scanned documents really look like: skewed, faded, photographed at an angle.
  The human labels give the Session 1 labelling exercise in notebook 02 a reference to measure annotator agreement against,
  and the extraction evaluation a golden set that nobody on the team wrote.
- **SBA SOP 50 10** is a real rulebook with real section numbering, cross-references and legal phrasing.
  Chunking and retrieval that look fine on a tidy 40-page synthetic manual fall apart on a document
  like this, which is the point of the Session 2 retrieval bake-off. The Good Bank manual stays as the source for
  policy checks, because its thresholds are what the answer key was generated from.
- **ChartQA** gives real charts with known answers, so visual retrieval (ColPali) and a vision model can be
  scored rather than just demonstrated.
- **The SBA loan book** gives real default labels. Good Bank is a new lender with no history of its own,
  so it bootstraps a risk model from the public loan book. That is exactly what an FDE does when the
  client has no labels yet, and the adapter that maps a Good Bank application onto the model's schema
  is where train/serve skew is taught. Business names, cities and bank names are dropped at ingest.

## Things to tell participants

- The receipts and charts are not Rohan's documents. In the story they stand in for supplier invoices
  and management charts that an applicant attaches to a file.
- The SBA rulebook is American and the loan book is in US dollars. The story is Indian (INR, PAN, UPI).
  The mismatch is deliberate and discussed in Session 2 (notebooks 06 and 07): a model trained on one population and applied to
  another needs an adapter, a fairness check and a drift monitor.
- Check each free-tier provider's data-use terms before sending anything. Only synthetic or public data
  goes to an LLM in this project.
- Licences: CC BY-SA 4.0 (SBA loan book) requires attribution and share-alike if you redistribute a
  derivative. GPL 3.0 applies to the ChartQA dataset files. Keep the samples out of the repo (they are),
  and keep this attribution page with any copy of the project.
