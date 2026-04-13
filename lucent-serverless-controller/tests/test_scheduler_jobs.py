"""Tests for scheduler job-queue-aware scaling logic."""

from __future__ import annotations

from unittest import mock

import pytest

from lucent_serverless_controller import db
from lucent_serverless_controller.scheduler import _tick


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn_singleton", None)
    db.init_db()
    yield
    if db._conn_singleton:
        db._conn_singleton.close()
        db._conn_singleton = None


def _register_app(app_id: str = "job-app", mode: str = "job", min_c: int = 0, max_c: int = 4):
    db.upsert_app(
        app_id, "test:latest",
        mode=mode, compute_type="CPU", cpu_flavor="cpu3c",
        min_concurrency=min_c, max_concurrency=max_c,
    )


class TestJobScaleUp:
    @mock.patch("lucent_serverless_controller.scheduler.spawn_pod")
    def test_spawns_pod_when_jobs_pending(self, mock_spawn):
        mock_spawn.return_value = "pod-1"
        _register_app()
        db.insert_job("j1", "job-app", '{}')

        _tick()

        mock_spawn.assert_called_once()
        call_args = mock_spawn.call_args[0][0]
        assert call_args["app_id"] == "job-app"

    @mock.patch("lucent_serverless_controller.scheduler.spawn_pod")
    def test_no_spawn_when_queue_empty(self, mock_spawn):
        _register_app()
        _tick()
        mock_spawn.assert_not_called()

    @mock.patch("lucent_serverless_controller.scheduler.spawn_pod")
    def test_respects_max_concurrency(self, mock_spawn):
        _register_app(max_c=2)
        # Insert 5 pending jobs
        for i in range(5):
            db.insert_job(f"j{i}", "job-app", '{}')
        # Simulate 2 existing pods
        db.insert_pod("p1", "job-app", "http://p1")
        db.insert_pod("p2", "job-app", "http://p2")

        _tick()

        mock_spawn.assert_not_called()

    @mock.patch("lucent_serverless_controller.scheduler.spawn_pod")
    def test_min_concurrency_still_works_for_realtime(self, mock_spawn):
        mock_spawn.return_value = "pod-1"
        _register_app("rt-app", mode="realtime", min_c=1)

        _tick()

        mock_spawn.assert_called_once()

    @mock.patch("lucent_serverless_controller.scheduler.spawn_pod")
    def test_expires_stale_jobs_each_tick(self, mock_spawn):
        _register_app()
        db.insert_job("j1", "job-app", '{}', ttl_sec=1)
        db.dequeue_job("job-app")

        # Backdate so it's past TTL
        import time
        with db._write_lock:
            db._conn().execute(
                "UPDATE jobs SET started_at = ? WHERE job_id = ?",
                (int(time.time()) - 10, "j1"),
            )
            db._conn().commit()

        _tick()

        job = db.get_job("j1")
        assert job["status"] == "failed"
        assert job["error"] == "timeout"
