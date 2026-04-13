"""Tests for the jobs API endpoints using FastAPI TestClient.

No RunPod dependency — tests run entirely against the in-process app.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from lucent_serverless_controller import db
from lucent_serverless_controller.main import create_app


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn_singleton", None)
    monkeypatch.delenv("LUCENT_API_KEY", raising=False)
    db.init_db()
    yield
    if db._conn_singleton:
        db._conn_singleton.close()
        db._conn_singleton = None


@pytest.fixture()
def client(_fresh_db):
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register_job_app(client: TestClient, app_id: str = "test-job-app"):
    resp = client.post("/apps", json={
        "app_id": app_id,
        "image_ref": "test:latest",
        "mode": "job",
        "compute_type": "CPU",
    })
    assert resp.status_code == 200


# ── POST /run ─────────────────────────────────────────────────────────

class TestSubmitJob:
    def test_submit_returns_job_id(self, client):
        _register_job_app(client)
        resp = client.post("/run", json={
            "app_id": "test-job-app",
            "input": {"prompt": "hello"},
        })
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_submit_unknown_app(self, client):
        resp = client.post("/run", json={
            "app_id": "nope",
            "input": {},
        })
        assert resp.status_code == 404

    def test_submit_to_realtime_app_fails(self, client):
        client.post("/apps", json={
            "app_id": "rt-app",
            "image_ref": "test:latest",
            "mode": "realtime",
        })
        resp = client.post("/run", json={
            "app_id": "rt-app",
            "input": {},
        })
        assert resp.status_code == 400
        assert "realtime" in resp.json()["detail"].lower()


# ── GET /status/{job_id} ─────────────────────────────────────────────

class TestJobStatus:
    def test_status_pending(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {"x": 1},
        }).json()["job_id"]

        resp = client.get(f"/status/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    def test_status_unknown_job(self, client):
        resp = client.get("/status/nope")
        assert resp.status_code == 404


# ── GET /jobs/next (worker dequeue) ──────────────────────────────────

class TestDequeue:
    def test_dequeue_returns_job(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {"val": 42},
        }).json()["job_id"]

        resp = client.get("/jobs/next", params={
            "worker_id": "w1", "app_id": "test-job-app",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["input"] == {"val": 42}

    def test_dequeue_empty(self, client):
        _register_job_app(client)
        resp = client.get("/jobs/next", params={
            "worker_id": "w1", "app_id": "test-job-app",
        })
        assert resp.status_code == 204


# ── POST /jobs/{id}/result ───────────────────────────────────────────

class TestPostResult:
    def test_complete_job(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {},
        }).json()["job_id"]
        client.get("/jobs/next", params={
            "worker_id": "w1", "app_id": "test-job-app",
        })

        resp = client.post(f"/jobs/{job_id}/result", json={
            "output": {"image": "base64..."},
        })
        assert resp.status_code == 200

        status = client.get(f"/status/{job_id}").json()
        assert status["status"] == "completed"
        assert status["output"] == {"image": "base64..."}

    def test_fail_job(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {},
        }).json()["job_id"]
        client.get("/jobs/next", params={
            "worker_id": "w1", "app_id": "test-job-app",
        })

        resp = client.post(f"/jobs/{job_id}/result", json={
            "error": "OOM",
        })
        assert resp.status_code == 200

        status = client.get(f"/status/{job_id}").json()
        assert status["status"] == "failed"
        assert status["error"] == "OOM"

    def test_result_on_pending_job_rejected(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {},
        }).json()["job_id"]

        resp = client.post(f"/jobs/{job_id}/result", json={"output": {}})
        assert resp.status_code == 409


# ── POST /cancel/{job_id} ────────────────────────────────────────────

class TestCancel:
    def test_cancel_pending(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {},
        }).json()["job_id"]

        resp = client.post(f"/cancel/{job_id}")
        assert resp.status_code == 200

        status = client.get(f"/status/{job_id}").json()
        assert status["status"] == "cancelled"

    def test_cancel_running_fails(self, client):
        _register_job_app(client)
        job_id = client.post("/run", json={
            "app_id": "test-job-app", "input": {},
        }).json()["job_id"]
        client.get("/jobs/next", params={
            "worker_id": "w1", "app_id": "test-job-app",
        })

        resp = client.post(f"/cancel/{job_id}")
        assert resp.status_code == 409


# ── GET /jobs (dashboard) ────────────────────────────────────────────

class TestListJobs:
    def test_list_returns_recent(self, client):
        _register_job_app(client)
        for i in range(3):
            client.post("/run", json={
                "app_id": "test-job-app", "input": {"i": i},
            })

        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 3
