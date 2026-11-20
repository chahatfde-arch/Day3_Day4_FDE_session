"""Milestone 4: turn a PDF or a photo into a validated Python object.

Three ideas live here, and they are the whole of document extraction:

1. **Read the page.** A "text PDF" already carries its characters, so reading it is
   free and perfect. A "scan" is only a picture of a page, so it has to go through
   OCR, which is slow and imperfect. `creditlens.extraction.ocr` hides that difference.
2. **Ask the model for a shape, not for prose.** `creditlens.extraction.extract`
   sends the page text to the gateway together with a Pydantic model, so what comes
   back is either a valid object or a loud error. Never a paragraph to parse by hand.
3. **Measure it.** `creditlens.extraction.golden` builds the golden set and
   `evals/extraction_accuracy.py` scores the extractor against it, field by field.
"""
from creditlens.extraction.extract import (extract_document, extract_financials,  # noqa: F401
                                           extract_receipt)
from creditlens.extraction.ocr import OCRUnavailable, image_to_text, pdf_to_text  # noqa: F401

__all__ = ["OCRUnavailable", "pdf_to_text", "image_to_text",
           "extract_document", "extract_financials", "extract_receipt"]
