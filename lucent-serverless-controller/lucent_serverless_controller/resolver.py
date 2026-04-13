"""GET /resolve — find a warm pod or cold-start one for the requested app.

Also hosts app and pod CRUD endpoints for visibility and registration.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lucent_serverless.runpod_client import (
    RunpodError,
    pod_proxy_url,
    terminate_pod,
)

from . import db
from .scheduler import spawn_pod

log = logging.getLogger(__name__)
router = APIRouter()


# ── /resolve ──────────────────────────────────────────────────────────
#
# Non-blocking: always returns immediately. The controller's scheduler
# loop promotes pending pods to ready when their /health comes up, so
# the client just polls /resolve until status == "ready".
#
# This dodges RunPod's ~100s Cloudflare proxy timeout.

@router.get("/resolve")
def resolve(app_id: str):
    app = db.get_app(app_id)
    if not app:
        raise HTTPException(404, f"unknown app_id {app_id!r}")

    # 1. Warm pod with room → return immediately.
    for pod in db.pods_by_app_and_status(app_id, ["ready"]):
        if pod["active_connections"] < app["max_concurrency"]:
            return {"status": "ready", "pod_url": pod["url"], "pod_id": pod["pod_id"]}

    # 2. Pending pod already booting → tell client to keep waiting.
    pending = db.pods_by_app_and_status(app_id, ["pending"])
    if pending:
        pod = pending[0]
        return {
            "status": "pending",
            "pod_url": pod["url"],
            "pod_id": pod["pod_id"],
        }

    # 3. Nothing alive → spawn and return pending.
    log.info("no warm or pending pod for %s, cold-starting", app_id)
    try:
        pod_id = spawn_pod(dict(app))
    except RunpodError as e:
        raise HTTPException(502, f"failed to spawn pod: {e}")
    url = pod_proxy_url(pod_id)
    return {"status": "pending", "pod_url": url, "pod_id": pod_id}


# ── App CRUD ──────────────────────────────────────────────────────────

class AppRegistration(BaseModel):
    app_id: str
    image_ref: str
    compute_type: str = "GPU"
    machine_type: str = "NVIDIA GeForce RTX 4090"
    cpu_flavor: str = "cpu3c"
    vcpu_count: int = 2
    min_concurrency: int = 0
    max_concurrency: int = 4
    keep_alive_sec: int = 300
    container_disk_gb: int = 40
    cloud_type: str = "SECURE"
    env: dict[str, str] | None = None


@router.post("/apps")
def register_app(body: AppRegistration):
    db.upsert_app(
        body.app_id,
        body.image_ref,
        compute_type=body.compute_type,
        machine_type=body.machine_type,
        cpu_flavor=body.cpu_flavor,
        vcpu_count=body.vcpu_count,
        min_concurrency=body.min_concurrency,
        max_concurrency=body.max_concurrency,
        keep_alive_sec=body.keep_alive_sec,
        container_disk_gb=body.container_disk_gb,
        cloud_type=body.cloud_type,
        env=body.env,
    )
    log.info("registered app %s -> %s", body.app_id, body.image_ref)
    return {"ok": True, "app_id": body.app_id}


@router.get("/apps")
def list_apps():
    return [dict(row) for row in db.list_apps()]


@router.get("/apps/{app_id}")
def get_app(app_id: str):
    app = db.get_app(app_id)
    if not app:
        raise HTTPException(404, f"unknown app_id {app_id!r}")
    return dict(app)


@router.delete("/apps/{app_id}")
def remove_app(app_id: str):
    # Terminate any live RunPod pods first so we don't leak compute.
    live = db.pods_by_app_and_status(app_id, ["pending", "ready"])
    terminated: list[str] = []
    failed: list[str] = []
    for pod in live:
        try:
            terminate_pod(pod["pod_id"])
            terminated.append(pod["pod_id"])
        except RunpodError as e:
            log.warning("failed to terminate pod %s: %s", pod["pod_id"], e)
            failed.append(pod["pod_id"])

    if not db.delete_app(app_id):
        raise HTTPException(404, f"unknown app_id {app_id!r}")
    log.info(
        "deleted app %s (terminated=%d, failed=%d)",
        app_id, len(terminated), len(failed),
    )
    return {"ok": True, "terminated": terminated, "failed_to_terminate": failed}


# ── Pod visibility ────────────────────────────────────────────────────

@router.get("/pods")
def list_pods():
    return [dict(row) for row in db.live_pods()]


# ── Pod history ──────────────────────────────────────────────────────

@router.get("/pod-history")
def pod_history(app_id: str | None = None, limit: int = 50):
    return [dict(row) for row in db.list_pod_history(app_id=app_id, limit=limit)]
