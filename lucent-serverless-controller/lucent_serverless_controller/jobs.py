"""Job-based serverless endpoints: /run, /runsync, /status, /cancel.

Also exposes worker-facing endpoints for pulling work and posting results:
  GET  /jobs/next?worker_id=&app_id=
  POST /jobs/{job_id}/result
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from . import db
from .webhook import deliver_webhook

log = logging.getLogger(__name__)
router = APIRouter()


# ── Client-facing ─────────────────────────────────────────────────────

class RunRequest(BaseModel):
    app_id: str
    input: dict
    webhook_url: str | None = None
    ttl_sec: int = 300


@router.post("/run")
def submit_job(body: RunRequest):
    app = db.get_app(body.app_id)
    if not app:
        raise HTTPException(404, f"unknown app_id {body.app_id!r}")
    if app["mode"] != "job":
        raise HTTPException(
            400,
            f"app {body.app_id!r} is mode={app['mode']!r}, not 'job'. "
            "Use /resolve for realtime apps.",
        )

    job_id = uuid.uuid4().hex
    db.insert_job(
        job_id,
        body.app_id,
        json.dumps(body.input),
        webhook_url=body.webhook_url,
        ttl_sec=body.ttl_sec,
    )
    log.info("enqueued job %s for app %s", job_id, body.app_id)
    return {"job_id": job_id}


@router.post("/runsync")
def submit_job_sync(body: RunRequest):
    """Submit and block until the job completes or times out."""
    app = db.get_app(body.app_id)
    if not app:
        raise HTTPException(404, f"unknown app_id {body.app_id!r}")
    if app["mode"] != "job":
        raise HTTPException(
            400,
            f"app {body.app_id!r} is mode={app['mode']!r}, not 'job'.",
        )

    job_id = uuid.uuid4().hex
    db.insert_job(
        job_id,
        body.app_id,
        json.dumps(body.input),
        webhook_url=body.webhook_url,
        ttl_sec=body.ttl_sec,
    )
    log.info("enqueued sync job %s for app %s", job_id, body.app_id)

    timeout = min(body.ttl_sec, 120)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = db.get_job(job_id)
        if job and job["status"] in ("completed", "failed", "cancelled"):
            return _job_response(job)
        time.sleep(0.5)

    raise HTTPException(408, f"job {job_id} did not complete within {timeout}s")


@router.get("/status/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"unknown job_id {job_id!r}")
    return _job_response(job)


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    if not db.cancel_job(job_id):
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(404, f"unknown job_id {job_id!r}")
        raise HTTPException(
            409, f"job {job_id} is {job['status']!r}, can only cancel pending jobs"
        )
    return {"ok": True, "job_id": job_id}


# ── Worker-facing ─────────────────────────────────────────────────────

@router.get("/jobs/next")
def dequeue_next(worker_id: str, app_id: str, response: Response):
    """Atomically claim the next pending job for the given app."""
    job = db.dequeue_job(app_id)
    if job is None:
        response.status_code = 204
        return None

    log.info("assigned job %s to worker %s", job["job_id"], worker_id)
    return {
        "job_id": job["job_id"],
        "input": json.loads(job["input"]),
    }


class JobResult(BaseModel):
    output: dict | None = None
    error: str | None = None


@router.post("/jobs/{job_id}/result")
def post_result(job_id: str, body: JobResult):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"unknown job_id {job_id!r}")
    if job["status"] != "running":
        raise HTTPException(409, f"job {job_id} is {job['status']!r}, expected 'running'")

    if body.error:
        db.fail_job(job_id, body.error)
        log.info("job %s failed: %s", job_id, body.error)
    else:
        db.complete_job(job_id, json.dumps(body.output))
        log.info("job %s completed", job_id)

    if job["webhook_url"]:
        updated = db.get_job(job_id)
        deliver_webhook(job["webhook_url"], _job_response(updated))

    return {"ok": True}


# ── Jobs visibility (for dashboard) ──────────────────────────────────

@router.get("/jobs")
def list_jobs(limit: int = 50):
    return [dict(row) for row in db.recent_jobs(limit)]


# ── Helpers ───────────────────────────────────────────────────────────

def _job_response(job) -> dict:
    resp: dict = {
        "job_id": job["job_id"],
        "app_id": job["app_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }
    if job["status"] == "completed" and job["output"]:
        resp["output"] = json.loads(job["output"])
    if job["status"] == "failed" and job["error"]:
        resp["error"] = job["error"]
    return resp
