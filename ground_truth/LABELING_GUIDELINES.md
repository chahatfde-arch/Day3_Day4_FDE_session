# Labeling guidelines: CreditLens AI document extraction

You are writing down the right answer for a handful of documents. That list of right
answers is the **golden set**, and everything in the Evaluate phase of
Build → Deploy → Evaluate → Maintain is measured against it. If the golden set is wrong,
every accuracy number after it is a lie that is expensive to find later.

Two rules sit above all the others:

1. **Copy what is printed.** You are a transcriber, not an analyst. If the paper says
   something odd, write the odd thing down.
2. **When you are unsure, do not guess silently.** Put the row in the `notes` column and
   raise it at the party (see the last section). A documented disagreement is useful; an
   invented number is not.

You will label two very different kinds of document, so read the section you need.

---

## Part 1: financial statements (Rohan's company accounts)

One row per **company and fiscal year**. A three-year statement gives you three rows.

| Column | What to write | Where to find it |
| --- | --- | --- |
| `company_id` | The id in the file name, e.g. `C003` | file name |
| `fiscal_year` | The column heading exactly as printed, e.g. `FY2026` | top of the table |
| `revenue` | Revenue from operations | profit and loss |
| `ebitda` | EBITDA | profit and loss |
| `interest` | Finance costs | profit and loss |
| `pat` | Profit after tax | profit and loss |
| `current_assets` | Total current assets | balance sheet |
| `current_liabilities` | Trade payables and other current liabilities | balance sheet |
| `total_debt` | Total borrowings | balance sheet |
| `principal_repayment` | Scheduled principal repayment **for that year** | Note 14, Borrowings |
| `notes` | Anything that made you hesitate | you |

### How to write the numbers

- **Units.** The statement is printed in INR thousands and the heading says so:
  `(INR '000)`. Write the number **as printed**. Do not convert to rupees, lakhs or
  crores. `52,316` on the page is `52316` in the sheet.
- **Commas.** Indian digit grouping puts commas every two digits after the first three:
  `12,34,567`. Strip every comma. That is `1234567`.
- **Brackets mean negative.** Accountants print a loss as `(4,120)`. Write `-4120`.
  A minus sign means the same thing; write it the same way.
- **Rounding.** Never round, never recalculate, never annualise. If the page and your
  calculator disagree, the page wins and you write a note.
- **Decimals.** These statements print whole thousands. If you ever see a decimal,
  keep it exactly as printed rather than rounding it away.
- **Missing values.** If the line item is not printed at all, leave the cell **empty**.
  Empty means "the document does not say". Do not write `0`, `NA`, `-` or `unknown`:
  zero is a claim about the business, and empty is a fact about the paper.

### Tie-break rules for statements

Pick the first rule that applies.

1. **The printed label wins over your judgement.** `interest` is whatever sits on the
   line called "Finance costs", even if you think some of it is a fee.
2. **The most specific label wins.** "Total current assets" is not "Total assets".
   "Total borrowings" is not the "Note 14: Borrowings" heading.
3. **The statement wins over the note, except for `principal_repayment`,** which only
   exists in Note 14. Take the amount for that fiscal year, not the closing balance.
4. **The table wins over the chart.** The bar chart on page 1 is a picture of the same
   numbers; never read a value off it.
5. **If the same line appears twice**, take the one inside the main table, not a
   subtotal repeated in a note.
6. **If the scan is unreadable**, leave the cell empty and write `illegible` in `notes`.
   Do not squint and guess. An illegible cell is a known gap; a guess is a silent error.

---

## Part 2: receipts (SROIE scans)

One row per image file. Four fields, all of them **strings**, all of them copied.

| Column | What to write |
| --- | --- |
| `file` | The image file name, e.g. `X00016469670.jpg` |
| `company` | The merchant name as printed, including SDN BHD / PVT LTD if printed |
| `date` | The transaction date **exactly as printed**: `15/01/2019`, `25 JAN 2018` |
| `address` | The merchant's street address, as one line |
| `total` | The grand total payable, digits and a decimal point only: `193.00` |
| `notes` | Anything that made you hesitate |

### How to write the fields

- **`company`.** Usually the largest text at the top. Exclude the registration number
  (`ROC NO: 538358-H`), the branch name and the address. Keep the spelling and the
  capitalisation on the paper, including typos.
- **`date`.** Copy the characters, do not reformat. `15/01/2019` stays `15/01/2019`;
  it does not become `2019-01-15`. Drop the time of day. If the receipt prints both an
  order date and a print date, take the one next to the invoice or bill number.
- **`address`.** Join the address lines into one line separated by commas. Leave out the
  phone number, fax, email and tax id. If no address is printed, leave it empty.
- **`total`.** The amount the customer actually pays. Strip the currency symbol and the
  thousands separator: `RM 1,193.00` becomes `1193.00`. Keep both decimal places.

### Tie-break rules for receipts

1. **`TOTAL` beats `SUBTOTAL`,** always.
2. **Rounded total beats net total.** If the receipt shows `TOTAL 38.85` and
   `ROUNDED TOTAL 38.90`, the customer pays 38.90, so write `38.90`.
3. **"Total inclusive GST" beats "Total exclusive GST".**
4. **Cash tendered and change are not the total.** `CASH 50.00 / CHANGE 11.10` means the
   total is 38.90, and 50.00 is a trap.
5. **If two totals are printed and neither is clearly the final one**, take the lower
   one and write a note. Over-stating what a borrower paid is the worse error.
6. **Unreadable field**: leave it empty and write `illegible`. Same rule as statements.

---

## The labeling party: five documents, one hour

Do this **before** labelling the whole set. The point is not to produce labels; it is to
find out where the guidelines are ambiguous while it is still cheap to fix them.

**Who.** Two labellers per document, plus one facilitator who does not label.

**The shared sheet.** One spreadsheet, one tab per labeller, identical columns, and a
third tab for decisions. Each labeller fills in their own tab and **cannot see the
other tab** until the comparison step. Columns are exactly the ones in the tables above,
in that order, with `notes` last. Export each tab as CSV when you are done, named
`labels_<yourname>.csv`.

**The hour.**

1. **5 min, set up.** Pick five documents: three text statements, one scanned statement,
   one receipt. Everyone opens the same five files.
2. **10 min, read the rules.** Read Part 1 or Part 2 out loud together, including the
   tie-break lists. Questions now are free.
3. **20 min, label alone.** No talking, no peeking. Silence is the whole experiment:
   you are testing the guidelines, not the people.
4. **10 min, measure.** The facilitator runs:

       python ground_truth/agreement.py labels_a.csv labels_b.csv --key file

   and reads out the per-field table. Expect `total` and `revenue` to be near 1.0 and
   `address` and `company` to be the weak ones.
5. **10 min, resolve.** Go through every disagreement. For each one, decide the right
   answer **and** write the rule that made it right. If no existing rule decides it,
   add a new tie-break line to this file. That new line is the real output of the party.
6. **5 min, agree the bar.** Mean kappa above 0.8 means label the rest. Below 0.6 means
   the instructions are broken: fix them and run a second party on five new documents.

**Afterwards.** The agreed labels go into `ground_truth/golden_set.json`, the updated
rules go into this file, and both are committed in the same pull request so a reviewer
can see which rule change moved which label. Never edit the golden set without saying
why in the commit message.

---

## Why this is worth an hour

The model can be swapped in a minute and the prompt in ten seconds. The golden set is
the only part of the extraction pipeline that cannot be regenerated, and
`evals/extraction_accuracy.py` can only ever be as trustworthy as the labels underneath
it. Everything downstream — the risk score, the credit memo, the exception that Priya
has to justify to an underwriter — inherits the care you take here.
