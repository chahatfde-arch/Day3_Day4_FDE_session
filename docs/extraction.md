# Document extraction (Day 2)

*Beginner notes for milestone 4. Code: `src/creditlens/extraction/`, `ground_truth/`, `evals/extraction_accuracy.py`.*

## The problem in one sentence

Rohan's file has numbers trapped in PDFs and photos, and Priya needs them as typed, checked values.

## Text PDFs versus scans

A PDF can carry its text inside the file (a "text PDF", made by software) or be nothing but pictures
of pages (a "scan", made by a scanner or a phone). You cannot tell from the file name.

| | Text PDF | Scanned PDF or photo |
| --- | --- | --- |
| How we read it | `pypdf` pulls the characters out, milliseconds | render each page to an image, run OCR, seconds per page |
| Typical mistakes | none in the characters, but table columns may run together | `1,23,456` becomes `1,23,4S6`; rotated pages read as garbage |
| In our data | three quarters of the financial statements | one quarter, plus every receipt |

`pdf_to_text(path)` tries the text route first. If fewer than about 200 characters come back, the file
is a scan and OCR takes over. The function returns `(text, ocr_used)` so the extraction can record
which path was taken; that flag ends up in the memo's audit trail.

**OCR engines.** Tesseract is the classic, and is used when its binary is installed. RapidOCR is a
pure-pip alternative that works on any laptop and in Colab without `apt`. Both are free.

**Rotated pages.** The Day 2 curveball. We OCR a page at 0, 90, 180 and 270 degrees and keep the
orientation whose output contains the most real words (a short vocabulary like "revenue", "assets",
"balance" plus the share of alphabetic tokens). Crude, fast, and it works.

**Caching.** OCR output is cached under `data/cache/ocr/` by the file's hash. The second run of a
notebook is instant, which matters in a live room.

## From text to typed values: Pydantic and the LLM

We never ask the model for "the numbers". We hand it a JSON schema generated from a Pydantic model
and demand an answer that fits. Three ideas:

1. **A schema is a contract.** `FiscalYearFigures` says `revenue` is an optional integer in INR
   thousands. If the model answers with a string, validation fails and we know before the number
   reaches anyone.
2. **A discriminated union handles two document types with one call.** `ExtractionEnvelope.document`
   is either a `FinancialStatementExtraction` or a `ReceiptExtraction`; the `doc_type` field says
   which. The prompt tells the model which type to return.
3. **Retries on parse errors.** When validation fails, the gateway sends the error text back to the
   model and asks again, up to three times. Most failures are fixed on the second try.

All of this happens inside `creditlens.gateway.llm`, so the extraction code is twenty lines.

## Offline mode

With `LLM_OFFLINE=1`, a rule-based parser stands in for the model: it finds the printed line items
("Revenue from operations", "Total borrowings", the Note 14 sentence) with regular expressions and
reads the Indian-grouped numbers. On text PDFs it scores above 90 percent per field, so the workshop
runs even when the room has no quota. On receipts it is weaker (company names are hard without a
model), which is a good discussion: where does a rule stop and a model start?

## Ground truth before the extractor

`ground_truth/golden_set.json` holds the answers: ten financial statements from the generator's truth
and forty real receipts with human labels from the SROIE dataset. `LABELING_GUIDELINES.md` says how
to label and how to settle disagreements, and `agreement.py` measures Cohen's kappa between two
labellers. The point of the labelling party is not the labels; it is discovering that two careful
people disagree, and writing the rule that settles it.

## Running the evaluation

```bash
python -m creditlens.extraction.golden            # build ground_truth/golden_set.json
python -m evals.extraction_accuracy --limit 10    # per-field accuracy, writes evals/results/
python -m evals.extraction_accuracy --doc-types receipt
```

Strings are compared after normalising case, whitespace and punctuation; numbers must be within one
percent; dates are parsed before comparison. The overall score is the metric the CI gate watches.

## Free-tier reminders

Provider limits change often and are small. The gateway rate-limits, caches every answer on disk and
falls back from Gemini to Groq to a local Ollama model. Check each provider's page before a session,
set `LLM_RPM` accordingly, and ship the `data/cache/llm/` folder to the room as a backup.
