"""Milestone 4 done-when: text PDFs, a sideways scan, the golden set and the scorer.

Everything here runs with `LLM_OFFLINE=1`, no API key and no network: the gateway's
offline handler registered in `creditlens.extraction.extract` does the extracting.
The only optional part is OCR, which is skipped when no engine is installed.
"""
import json

import pytest

from creditlens.config import settings as default_settings
from creditlens.extraction.extract import extract_financials
from creditlens.extraction.golden import FIGURE_FIELDS, RECEIPT_FIELDS, build_golden_set
from creditlens.extraction.ocr import ocr_available, pdf_to_text
from creditlens.gateway.llm import get_gateway
from evals.extraction_accuracy import score

CHECKED_FIELDS = ("revenue", "ebitda", "total_debt")


@pytest.fixture(scope="module", autouse=True)
def offline_gateway():
    """Force offline mode and switch the shared cache off, so the tests prove the stub."""
    gateway = get_gateway()
    was_offline, cache = gateway.offline, gateway.cache
    gateway.offline, gateway.cache = True, None
    yield gateway
    gateway.offline, gateway.cache = was_offline, cache


@pytest.fixture(scope="module")
def doc_kinds(s):
    return json.loads((s.paths.answer_key / "financial_doc_kinds.json").read_text())


def _first_of_kind(doc_kinds, kind):
    matches = [company_id for company_id, k in doc_kinds.items() if k == kind]
    if not matches:
        pytest.skip(f"no {kind} PDF in the generated data")
    return sorted(matches)[0]


def _pdf(s, company_id):
    return s.paths.raw / "financials" / f"{company_id}_financials.pdf"


# ---------------------------------------------------------------------------
# Reading the page
# ---------------------------------------------------------------------------


def test_text_pdf_is_read_without_ocr(s, doc_kinds):
    """A PDF written by software already contains its characters: OCR must not be used."""
    import pandas as pd

    company_id = _first_of_kind(doc_kinds, "text")
    names = pd.read_csv(s.paths.raw / "registry" / "companies.csv").set_index("company_id")
    text, ocr_used = pdf_to_text(_pdf(s, company_id), s)

    assert ocr_used is False
    assert names.loc[company_id, "legal_name"] in text
    assert "Revenue from operations" in text


@pytest.mark.skipif(not ocr_available(), reason="no OCR engine installed")
def test_rotated_scan_is_turned_the_right_way_up(s, doc_kinds):
    """The curveball: one scan has page 2 rotated 90 degrees. OCR must still read it."""
    company_id = _first_of_kind(doc_kinds, "scan_rotated")
    text, ocr_used = pdf_to_text(_pdf(s, company_id), s)

    assert ocr_used is True
    assert "balance sheet" in text.lower()


# ---------------------------------------------------------------------------
# Extracting the numbers
# ---------------------------------------------------------------------------


def test_offline_extraction_matches_the_answer_key(s, doc_kinds):
    """Three text PDFs, every figure within 1 percent of the truth we printed them from."""
    truth = json.loads((s.paths.answer_key / "financials_truth.json").read_text())
    text_ids = sorted(c for c, kind in doc_kinds.items() if kind == "text")[:3]
    assert len(text_ids) == 3

    for company_id in text_ids:
        document = extract_financials(company_id, s)
        assert document.ocr_used is False
        assert document.source_file == f"{company_id}_financials.pdf"
        assert len(document.years) == 3

        for year in document.years:
            expected = truth[company_id]["years"][year.fiscal_year]
            for field in CHECKED_FIELDS:
                actual = getattr(year, field)
                assert actual is not None, f"{company_id} {year.fiscal_year} {field} missing"
                assert abs(actual - expected[field]) <= abs(expected[field]) * 0.01, (
                    f"{company_id} {year.fiscal_year} {field}: {actual} vs {expected[field]}")


# ---------------------------------------------------------------------------
# The golden set
# ---------------------------------------------------------------------------


def test_golden_set_has_both_lists(s, tmp_path):
    out = build_golden_set(s, out_path=tmp_path / "golden_set.json")
    golden = json.loads(out.read_text(encoding="utf-8"))

    assert isinstance(golden["financial_statements"], list)
    assert isinstance(golden["receipts"], list)
    assert len(golden["financial_statements"]) == 10

    entry = golden["financial_statements"][0]
    assert entry["company_id"] and entry["source_file"].endswith(".pdf")
    figures = next(iter(entry["years"].values()))
    assert set(FIGURE_FIELDS) <= set(figures)


def test_golden_set_includes_the_real_receipts(tmp_path):
    """Receipts come from the SROIE sample under data/real, outside the temp data root."""
    labels = default_settings.paths.real / "receipts" / "labels.json"
    if not labels.exists():
        pytest.skip("SROIE receipts not downloaded (python -m creditlens.data_real)")

    out = build_golden_set(default_settings, out_path=tmp_path / "golden_set.json")
    golden = json.loads(out.read_text(encoding="utf-8"))

    assert golden["receipts"]
    assert set(golden["receipts"][0]["entities"]) == set(RECEIPT_FIELDS)


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


def test_scorer_gives_one_for_identical_predictions():
    golden = {"C003:FY2026": {"company_name": "Meru Traders Pvt Ltd", "revenue": 72052},
              "receipt:a.jpg": {"company": "OJC MARKETING SDN BHD", "date": "15/01/2019",
                                "total": "193.00"}}
    result = score({k: dict(v) for k, v in golden.items()}, golden)

    assert result["overall"]["accuracy"] == 1.0
    assert all(bucket["accuracy"] == 1.0 for bucket in result["per_field"].values())
    assert result["records_without_prediction"] == []


def test_scorer_normalises_and_tolerates():
    """Punctuation and case do not matter, 1 percent on numbers does not matter, dates parse."""
    golden = {"r": {"company": "OJC Marketing Sdn Bhd.", "revenue": 100000, "date": "15/01/2019"}}
    predictions = {"r": {"company": "OJC  MARKETING SDN BHD", "revenue": 100500, "date": "15-01-2019"}}

    assert score(predictions, golden)["overall"]["accuracy"] == 1.0

    too_far = {"r": {"company": "OJC Marketing Sdn Bhd.", "revenue": 102000, "date": "15/01/2019"}}
    assert score(too_far, golden)["per_field"]["revenue"]["accuracy"] == 0.0


def test_scorer_handles_missing_fields_and_missing_documents():
    golden = {"a": {"revenue": 100, "ebitda": 10, "address": ""},
              "b": {"revenue": 200, "ebitda": 20, "address": ""}}

    # "a" is predicted but ebitda is None; "b" was never extracted at all.
    result = score({"a": {"revenue": 100, "ebitda": None}}, golden)

    assert result["records_without_prediction"] == ["b"]
    assert "address" not in result["per_field"]  # a blank golden value is not scorable
    assert result["per_field"]["revenue"]["n"] == 2
    assert result["per_field"]["revenue"]["correct"] == 1
    assert result["per_field"]["ebitda"]["correct"] == 0
    assert result["overall"]["accuracy"] == pytest.approx(0.25)
