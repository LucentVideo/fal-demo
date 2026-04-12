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
    wait_until_ready,
)

from . import db
from .scheduler import spawn_pod

log = logging.getLogger(__name__)
router = APIRouter()

RESOLVE_TIMEOUT = 120.0


# ── /resolve ──────────────────────────────────────────────────────────

@router.get("/resolve")
def resolve(app_id: str):
    app = db.get_app(app_id)
    if not app:
        raise HTTPException(404, f"unknown app_id {app_id!r}")

    # Try to find a warm pod with room (pack hot: fullest first)
    candidates = db.pods_by_app_and_status(app_id, ["ready"])
    for pod in candidates:
        if pod["active_connections"] < app["max_concurrency"]:
            return {"pod_url": pod["url"]}

    # Cold start: spawn and wait
    log.info("no warm pod for %s, cold-starting", app_id)
    try:
        pod_id = spawn_pod(dict(app))
    except RunpodError as e:
        raise HTTPException(502, f"failed to spawn pod: {e}")

    url = pod_proxy_url(pod_id)
    try:
        wait_until_ready(pod_id, timeout=RESOLVE_TIMEOUT)
    except TimeoutError:
        raise HTTPException(503, f"pod {pod_id} did not become ready within {RESOLVE_TIMEOUT:.0f}s")
    except RunpodError as e:
        raise HTTPException(502, f"pod {pod_id} failed: {e}")

    db.promote_pod(pod_id, url)
    log.info("cold-started pod %s for %s", pod_id, app_id)
    return {"pod_url": url}


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
    if not db.delete_app(app_id):
        raise HTTPException(404, f"unknown app_id {app_id!r}")
    return {"ok": True}


# ── Pod visibility ────────────────────────────────────────────────────

@router.get("/pods")
def list_pods():
    return [dict(row) for row in db.live_pods()]
