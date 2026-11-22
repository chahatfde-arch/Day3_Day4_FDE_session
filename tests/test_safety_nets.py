"""Safety nets added after real incidents on 2026-11-22. Each test is the promise the incident broke.

1. An old notebook cell ran `python -m creditlens.generators --skip-pdfs` on a laptop and replaced the
   data with a copy that had no PDFs. Extraction, retrieval and the evaluation gate all failed.
2. The gateway looked in its cache before checking offline mode, so the "offline" evaluation gate graded
   with a live model's cached answers and reported a judge agreement it had not earned.
"""
import subprocess
import sys

from creditlens.gateway.llm import DiskCache, Gateway


def test_a_fast_run_refuses_to_delete_pdfs_that_already_exist(tmp_path):
    pdf = tmp_path / "raw" / "financials" / "C001_fy2025.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 a stand-in for a real statement")
    run = subprocess.run([sys.executable, "-m", "creditlens.generators", "--skip-pdfs", "--data-root", str(tmp_path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Refusing to run" in run.stdout and "untouched" in run.stdout
    assert pdf.exists(), "the PDF must survive"


def test_offline_mode_never_answers_from_a_cached_live_reply(tmp_path):
    cache = DiskCache(tmp_path)
    gateway = Gateway(offline=True, cache=cache)
    messages = [{"role": "user", "content": "In one sentence, what is a debt service coverage ratio?"}]
    cache.put(DiskCache.key(gateway.models[0], messages, None),
              {"text": "an answer a live model gave yesterday", "model": "gemini/a-live-model"})
    reply = gateway.complete(messages)
    assert reply.model == "offline-stub", "offline mode must not grade or answer with a live model's cached reply"
    assert "live model gave yesterday" not in reply.text
    assert gateway.cache_hits == 0
