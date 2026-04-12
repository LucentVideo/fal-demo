"""Tests for the jobs layer in db.py.

All tests use an in-memory SQLite database — no RunPod dependency.
"""

from __future__ import annotations

import json
import threading
import time
from unittest import mock

import pytest

from lucent_serverless_controller import db


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point db at a fresh temp file for every test."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn_singleton", None)
    db.init_db()
    yield
    if db._conn_singleton:
        db._conn_singleton.close()
        db._conn_singleton = None


def _register_job_app(app_id: str = "test-app") -> None:
    db.upsert_app(
        app_id, "test-image:latest",
        mode="job", compute_type="CPU", cpu_flavor="cpu3c",
    )


# ── insert + get ──────────────────────────────────────────────────────

class TestInsertAndGet:
    def test_insert_and_retrieve(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{"prompt":"hello"}', ttl_sec=60)
        job = db.get_job("j1")
        assert job is not None
        assert job["status"] == "pending"
        assert json.loads(job["input"]) == {"prompt": "hello"}

    def test_get_nonexistent_returns_none(self):
        assert db.get_job("nope") is None


# ── dequeue ───────────────────────────────────────────────────────────

class TestDequeue:
    def test_dequeue_returns_oldest_pending(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{"n":1}')
        time.sleep(0.01)
        db.insert_job("j2", "test-app", '{"n":2}')

        job = db.dequeue_job("test-app")
        assert job is not None
        assert job["job_id"] == "j1"
        assert job["status"] == "running"

    def test_dequeue_empty_returns_none(self):
        _register_job_app()
        assert db.dequeue_job("test-app") is None

    def test_dequeue_skips_running(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        db.dequeue_job("test-app")
        assert db.dequeue_job("test-app") is None

    def test_dequeue_atomicity(self):
        """N threads all try to dequeue the same job — exactly one wins."""
        _register_job_app()
        db.insert_job("j-race", "test-app", '{}')

        results: list[str | None] = []
        lock = threading.Lock()

        def grab():
            job = db.dequeue_job("test-app")
            with lock:
                results.append(job["job_id"] if job else None)

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0] == "j-race"


# ── complete / fail / cancel ──────────────────────────────────────────

class TestJobTransitions:
    def test_complete_job(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        db.dequeue_job("test-app")
        db.complete_job("j1", '{"result": 42}')

        job = db.get_job("j1")
        assert job["status"] == "completed"
        assert json.loads(job["output"]) == {"result": 42}
        assert job["completed_at"] is not None

    def test_fail_job(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        db.dequeue_job("test-app")
        db.fail_job("j1", "out of memory")

        job = db.get_job("j1")
        assert job["status"] == "failed"
        assert job["error"] == "out of memory"

    def test_cancel_pending(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        assert db.cancel_job("j1") is True
        assert db.get_job("j1")["status"] == "cancelled"

    def test_cancel_running_fails(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        db.dequeue_job("test-app")
        assert db.cancel_job("j1") is False

    def test_cancel_nonexistent_fails(self):
        assert db.cancel_job("nope") is False


# ── pending_job_count ─────────────────────────────────────────────────

class TestPendingCount:
    def test_counts_only_pending(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        db.insert_job("j2", "test-app", '{}')
        db.insert_job("j3", "test-app", '{}')
        db.dequeue_job("test-app")

        assert db.pending_job_count("test-app") == 2

    def test_zero_when_empty(self):
        _register_job_app()
        assert db.pending_job_count("test-app") == 0


# ── expire_stale_jobs ─────────────────────────────────────────────────

class TestExpireStaleJobs:
    def test_expires_past_ttl(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}', ttl_sec=1)
        db.dequeue_job("test-app")

        # Backdate started_at so it's past TTL
        with db._write_lock:
            db._conn().execute(
                "UPDATE jobs SET started_at = ? WHERE job_id = ?",
                (int(time.time()) - 10, "j1"),
            )
            db._conn().commit()

        expired = db.expire_stale_jobs()
        assert expired == 1
        job = db.get_job("j1")
        assert job["status"] == "failed"
        assert job["error"] == "timeout"

    def test_does_not_expire_fresh(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}', ttl_sec=300)
        db.dequeue_job("test-app")

        expired = db.expire_stale_jobs()
        assert expired == 0


# ── recent_jobs ───────────────────────────────────────────────────────

class TestRecentJobs:
    def test_returns_newest_first(self):
        _register_job_app()
        db.insert_job("j1", "test-app", '{}')
        # Backdate j1 so j2 is clearly newer
        with db._write_lock:
            db._conn().execute(
                "UPDATE jobs SET created_at = created_at - 10 WHERE job_id = 'j1'"
            )
            db._conn().commit()
        db.insert_job("j2", "test-app", '{}')

        jobs = db.recent_jobs(limit=10)
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "j2"

    def test_respects_limit(self):
        _register_job_app()
        for i in range(5):
            db.insert_job(f"j{i}", "test-app", '{}')

        assert len(db.recent_jobs(limit=3)) == 3


# ── apps mode column ─────────────────────────────────────────────────

class TestAppsMode:
    def test_default_mode_is_realtime(self):
        db.upsert_app("rt-app", "img:latest")
        app = db.get_app("rt-app")
        assert app["mode"] == "realtime"

    def test_job_mode(self):
        db.upsert_app("job-app", "img:latest", mode="job")
        app = db.get_app("job-app")
        assert app["mode"] == "job"
