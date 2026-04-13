"""Scheduler loop: spawn pods for min_concurrency, scale up for queued
jobs, poll /health on live pods, and expire stale jobs.

Runs in its own thread, started by the lifespan in main.py.
"""

from __future__ import annotations

import json
import logging
import os
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
from .code_store import CODE_DIR

log = logging.getLogger(__name__)

POLL_INTERVAL = 5

# Before marking dead we wait much longer: GPU apps can take several
# minutes to boot (deps install + model downloads + setup warmup). During
# this boot grace window we ignore /health failures. After grace, a
# smaller threshold catches actually-unhealthy pods.
BOOT_GRACE_SEC = 600  # 10 minutes
MAX_FAILED_POLLS_AFTER_GRACE = 5


# ── Spawn ─────────────────────────────────────────────────────────────

def spawn_pod(app: dict) -> str:
    """Create a RunPod pod for the given app. Returns the RunPod pod id."""
    env = json.loads(app["env"] or "{}")
    env.setdefault("LUCENT_APP_ID", app["app_id"])

    # Tell the pod where the controller is (for wheel + code downloads)
    controller_url = os.environ.get("LUCENT_CONTROLLER_URL", "")
    if controller_url:
        env["LUCENT_CONTROLLER_URL"] = controller_url

        # If user code has been uploaded, tell the pod where to fetch it
        code_tar = CODE_DIR / f"{app['app_id']}.tar.gz"
        if code_tar.exists():
            env["LUCENT_CODE_URL"] = f"{controller_url}/apps/{app['app_id']}/code"

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
        registry_auth_id=os.environ.get("RUNPOD_REGISTRY_AUTH_ID") or None,
    )

    pod = create_pod(spec)
    pod_id = pod["id"]
    url = pod_proxy_url(pod_id)

    db.insert_pod(pod_id, app["app_id"], url)
    db.insert_pod_history(pod_id, app["app_id"])
    log.info("spawned pod %s for app %s", pod_id, app["app_id"])
    return pod_id


# ── Health polling ────────────────────────────────────────────────────

def _poll_pod(pod: dict) -> None:
    """Poll a single pod's /health and update the DB."""
    url = pod["url"] or pod_proxy_url(pod["pod_id"])

    age = int(time.time()) - pod["started_at"]
    in_grace = age < BOOT_GRACE_SEC

    def _maybe_mark_dead(reason: str) -> None:
        failed = db.increment_failed_polls(pod["pod_id"])
        if in_grace:
            # Still booting — don't kill. Log every 10 failed polls so
            # we can see it in the controller logs if it gets stuck.
            if failed % 10 == 0:
                log.info(
                    "pod %s still booting (%ds old, %d failed polls, %s)",
                    pod["pod_id"], age, failed, reason,
                )
            return
        if failed >= MAX_FAILED_POLLS_AFTER_GRACE:
            log.warning(
                "pod %s %s after grace period (%d polls), marking dead",
                pod["pod_id"], reason, failed,
            )
            db.set_pod_status(pod["pod_id"], "dead")

    try:
        with httpx.Client(timeout=5.0) as c:
            resp = c.get(f"{url}/health")
    except httpx.HTTPError:
        _maybe_mark_dead("unreachable")
        return

    if resp.status_code != 200:
        _maybe_mark_dead(f"unhealthy ({resp.status_code})")
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

    # Record boot timings into pod_history (once per pod)
    boot_timings = data.get("boot_timings")
    if boot_timings and not db.boot_timings_recorded(pod["pod_id"]):
        db.update_pod_boot_timings(pod["pod_id"], boot_timings)
        log.info("recorded boot timings for %s/%s: %.2fs",
                 pod["app_id"], pod["pod_id"], boot_timings.get("total_sec", 0))


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
