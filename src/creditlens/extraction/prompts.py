"""The words we send to the model, kept in one file so they can be reviewed and diffed.

A prompt is code. It has a version, it can regress, and a one-word change can move an
accuracy number, so it does not belong buried inside a function. Each prompt here does
four things, in this order, because that is the order a model pays attention in:

1. Says who it is and what the document is.
2. Says **which `doc_type` to return**. The schema is a discriminated union, so the
   model must put `"doc_type": "financial_statement"` or `"doc_type": "receipt"` in
   the answer or Pydantic cannot tell which half of the union to validate against.
3. Gives the rules a human labeller would be given (units, brackets, missing values).
4. Hands over the page text between markers, so the model knows where the document
   starts and the instructions stop.

The same markers are what the offline stub reads, which is why they are constants.
"""
from __future__ import annotations

# The offline stub (and any debugging human) finds the page text between these lines.
DOC_TEXT_START = "--- DOCUMENT TEXT START ---"
DOC_TEXT_END = "--- DOCUMENT TEXT END ---"
DOC_TYPE_LINE = "DOC_TYPE:"


FINANCIAL_SYSTEM = """You read audited financial statements for Good Bank and return structured data.

You are careful and literal. You never estimate, never annualise, never convert and never
recalculate a figure that the page already states. If a figure is not printed on the page,
you return null for it rather than guessing: a missing number is safe, a wrong number is not.

Return a JSON object whose "document" has "doc_type": "financial_statement"."""


FINANCIAL_USER = """Read this financial statement and return one entry per fiscal year.

Rules:
- The statement is printed in INR thousands. Report every amount as a whole number in
  INR thousands, exactly as printed: "12,34,567" becomes 1234567. Indian digit grouping
  puts commas every two digits after the first three; strip all commas.
- A figure in brackets, like (4,120), is negative: return -4120.
- Use the printed label, not your own idea of the line item:
    revenue              <- "Revenue from operations"
    ebitda               <- "EBITDA"
    interest             <- "Finance costs"
    total_debt           <- "Total borrowings"
    current_assets       <- "Total current assets"
    current_liabilities  <- "Trade payables and other current liabilities"
    pat                  <- "Profit after tax"
    principal_repayment  <- the scheduled principal repayment for that year in the
                            borrowings note (Note 14), not the closing balance
- fiscal_year is the column heading exactly as printed, for example "FY2026".
- company_name is the name printed at the top of the first page.
- unit is "INR thousands".
- If the page came from OCR the words may be imperfect. Match the nearest label and
  return null when a line is unreadable.
- confidence is your own 0 to 1 estimate that the figures are right.

{doc_type_line}
{start}
{text}
{end}"""


RECEIPT_SYSTEM = """You read scanned supplier receipts and invoices and return structured data.

You copy what is printed. You do not reformat dates, do not expand abbreviations, and do
not tidy up the merchant name. The value of this task is that the answer can be compared
to the paper.

Return a JSON object whose "document" has "doc_type": "receipt"."""


RECEIPT_USER = """Read this receipt and return the four key fields.

Rules:
- company: the merchant name as printed, usually the largest text on the first line.
  Keep the legal suffix (SDN BHD, PVT LTD) if it is printed. Do not include the address.
- date: the transaction date exactly as printed, for example "15/01/2019" or "25 JAN 2018".
  If several dates appear, use the one next to the invoice or bill number.
- address: the street address of the merchant as printed, joined into one line with commas.
  Return "" if no address is printed.
- total: the grand total actually payable, digits and a decimal point only, no currency
  symbol and no thousands separator: "RM 1,193.00" becomes "1193.00". If the receipt shows
  a subtotal, a tax line and a total, take the total. If it shows a rounded total and a net
  total, take the one labelled TOTAL.
- These are scans, so OCR noise is expected. Prefer the most plausible reading over null.
- confidence is your own 0 to 1 estimate that the fields are right.

{doc_type_line}
{start}
{text}
{end}"""


def build_messages(text: str, doc_type: str) -> list[dict]:
    """Assemble the chat messages for one document. The gateway adds the JSON schema."""
    if doc_type == "financial_statement":
        system, user = FINANCIAL_SYSTEM, FINANCIAL_USER
    elif doc_type == "receipt":
        system, user = RECEIPT_SYSTEM, RECEIPT_USER
    else:
        raise ValueError(f"unknown doc_type: {doc_type!r}")
    content = user.format(doc_type_line=f"{DOC_TYPE_LINE} {doc_type}",
                          start=DOC_TEXT_START, text=text, end=DOC_TEXT_END)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def document_text_from_messages(messages: list[dict]) -> tuple[str, str]:
    """Pull (page text, doc_type) back out of the messages. Used by the offline stub."""
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    body = user_messages[-1] if user_messages else ""
    doc_type = "financial_statement"
    for line in body.splitlines():
        if line.startswith(DOC_TYPE_LINE):
            doc_type = line[len(DOC_TYPE_LINE):].strip()
            break
    if DOC_TEXT_START in body:
        text = body.split(DOC_TEXT_START, 1)[1]
        text = text.split(DOC_TEXT_END, 1)[0]
    else:
        text = body
    return text.strip("\n"), doc_type
