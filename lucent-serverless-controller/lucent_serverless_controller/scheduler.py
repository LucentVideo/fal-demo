"""Scheduler loop: spawn pods for min_concurrency, scale up for queued
jobs, poll /health on live pods, and expire stale jobs.

Runs in its own thread, started by the lifespan in main.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import httpx

from lucent_serverless.runpod_client import (
    PodSpec,
    RunpodError,
    create_pod,
    pod_proxy_url,
)

from . import db

log = logging.getLogger(__name__)

POLL_INTERVAL = 5
MAX_FAILED_POLLS = 5


# ── Spawn ─────────────────────────────────────────────────────────────

def spawn_pod(app: dict) -> str:
    """Create a RunPod pod for the given app. Returns the RunPod pod id."""
    env = json.loads(app["env"] or "{}")
    env.setdefault("LUCENT_APP_ID", app["app_id"])

    spec = PodSpec(
        name=f"lucent-{app['app_id']}",
        image=app["image_ref"],
        compute_type=app["compute_type"],
        gpu_type_ids=(
            [app["machine_type"]] if app["compute_type"] == "GPU" else []
        ),
        cpu_flavor_ids=(
            [app["cpu_flavor"]] if app["compute_type"] == "CPU" else []
        ),
        vcpu_count=app["vcpu_count"],
        container_disk_gb=app["container_disk_gb"],
        cloud_type=app["cloud_type"],
        ports=["8000/http"],
        env=env,
    )

    pod = create_pod(spec)
    pod_id = pod["id"]
    url = pod_proxy_url(pod_id)

    db.insert_pod(pod_id, app["app_id"], url)
    log.info("spawned pod %s for app %s", pod_id, app["app_id"])
    return pod_id


# ── Health polling ────────────────────────────────────────────────────

def _poll_pod(pod: dict) -> None:
    """Poll a single pod's /health and update the DB."""
    url = pod["url"] or pod_proxy_url(pod["pod_id"])

    try:
        with httpx.Client(timeout=5.0) as c:
            resp = c.get(f"{url}/health")
    except httpx.HTTPError:
        failed = db.increment_failed_polls(pod["pod_id"])
        if failed >= MAX_FAILED_POLLS:
            log.warning("pod %s unreachable (%d polls), marking dead",
                        pod["pod_id"], failed)
            db.set_pod_status(pod["pod_id"], "dead")
        return

    if resp.status_code != 200:
        failed = db.increment_failed_polls(pod["pod_id"])
        if failed >= MAX_FAILED_POLLS:
            log.warning("pod %s unhealthy (%d polls), marking dead",
                        pod["pod_id"], failed)
            db.set_pod_status(pod["pod_id"], "dead")
        return

    data = resp.json()
    db.update_pod_health(
        pod["pod_id"],
        data.get("active_connections", 0),
        data.get("last_activity_at", int(time.time())),
    )
    if pod["status"] == "pending":
        db.promote_pod(pod["pod_id"], url)
        log.info("pod %s promoted to ready", pod["pod_id"])


# ── Loop ──────────────────────────────────────────────────────────────

def _tick() -> None:
    for app in db.list_apps():
        app_dict = dict(app)
        warm = db.pods_by_app_and_status(app["app_id"], ["pending", "ready"])

        # 1. Ensure min_concurrency for every app (realtime and job)
        deficit = app["min_concurrency"] - len(warm)
        for _ in range(deficit):
            try:
                spawn_pod(app_dict)
            except RunpodError as e:
                log.error("failed to spawn for %s: %s", app["app_id"], e)
                break

        # 2. For job-mode apps, scale up if there are pending jobs
        if app["mode"] == "job":
            pending = db.pending_job_count(app["app_id"])
            if pending > 0:
                idle_workers = sum(
                    1 for p in warm
                    if p["status"] == "ready" and p["active_connections"] == 0
                )
                needed = min(pending - idle_workers, app["max_concurrency"] - len(warm))
                for _ in range(max(0, needed)):
                    try:
                        spawn_pod(app_dict)
                    except RunpodError as e:
                        log.error("failed to scale up for %s: %s", app["app_id"], e)
                        break

    # 3. Health-poll every live pod
    for pod in db.live_pods():
        _poll_pod(dict(pod))

    # 4. Expire stale running jobs past their TTL
    expired = db.expire_stale_jobs()
    if expired:
        log.info("expired %d stale jobs", expired)


def scheduler_loop(stop: threading.Event) -> None:
    log.info("scheduler loop started (interval=%ds)", POLL_INTERVAL)
    while not stop.is_set():
        try:
            _tick()
        except Exception:
            log.exception("scheduler tick failed")
        stop.wait(POLL_INTERVAL)
    log.info("scheduler loop stopped")
