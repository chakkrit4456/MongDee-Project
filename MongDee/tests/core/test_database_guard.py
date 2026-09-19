"""Regression test for the production-DB guard added after a real incident:
an orphaned background test process wrote straight to data/mongdee.db and
permanently destroyed a real booth's rows. core.database now refuses to
touch that exact path while pytest is running (PYTEST_CURRENT_TEST is set
for the whole run), regardless of which function is called first.

This test necessarily calls into functions that *would* open
data/mongdee.db if the guard were broken — but the guard raises before
sqlite3.connect() is ever reached, so no file is created or touched even
if this assertion somehow failed to run.
"""

from __future__ import annotations

import os

import pytest

from core import database as db


def test_init_db_refuses_production_path():
    assert "PYTEST_CURRENT_TEST" in os.environ  # sanity: guard is actually active here
    with pytest.raises(RuntimeError, match="REFUSING TO RUN TESTS AGAINST PRODUCTION DATABASE"):
        db.init_db(db.DEFAULT_DB_PATH)


def test_connect_refuses_production_path():
    with pytest.raises(RuntimeError, match="REFUSING TO RUN TESTS AGAINST PRODUCTION DATABASE"):
        db._connect(db.DEFAULT_DB_PATH)


def test_isolated_tmp_path_is_unaffected(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)  # must not raise
    assert db_path.exists()
