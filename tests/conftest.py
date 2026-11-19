"""Test fixtures and isolation setup for DataFix test suite."""

import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path, monkeypatch):
    """Isolates SQLite audit ledger database and LLM cache directory to tmp_path

    so that running the test suite never leaves runtime files in the repository.
    """
    test_db = tmp_path / "test_audit_ledger.db"
    test_cache = tmp_path / "test_llm_cache"
    monkeypatch.setenv("DATAFIX_LEDGER_DB", str(test_db))
    monkeypatch.setenv("DATAFIX_LLM_CACHE_DIR", str(test_cache))
