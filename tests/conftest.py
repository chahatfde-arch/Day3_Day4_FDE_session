"""Shared fixtures: generate a fresh dataset once per test session in a temp folder."""
import os
from dataclasses import replace

import pytest

from creditlens.config import settings
from creditlens.generators.generate_all import generate_all
from creditlens.pipeline.run import run_pipeline

# Tests file tickets into their own JSON mock, whatever .env says: never into a real Jira site or the
# local Jira service. Empty, not deleted, so the MCP server processes the tests start (they inherit
# this environment and read .env with setdefault) stay on the mock too.
for _name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"):
    os.environ[_name] = ""


@pytest.fixture(scope="session")
def s(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    cfg = replace(settings, paths=settings.paths.with_data_root(root))
    generate_all(cfg, verbose=False)
    return cfg


@pytest.fixture(scope="session")
def outputs(s):
    return run_pipeline(s, verbose=False)
